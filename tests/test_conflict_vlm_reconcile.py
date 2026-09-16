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

"""Tests for Conflict-VLM compile-time fixes (G1b / G2 / G3).

These cover the deterministic reconciliation applied in
``conflict_vlm_materializer._compile_round_to_task``:

  * G3 - bare/timestamp twin de-duplication of the round's visible assets,
  * G2 - dropping ``assets_accessed`` checks whose asset is not visible,
  * G1(b) - augmenting the round instruction with an explicit asset-inspection
    step when an ``assets_accessed`` check exists.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.scenario.conflict_vlm_materializer import (
    _asset_base_key,
    _augment_instruction_for_asset_checks,
    _canonical_visible_asset_map,
    _dedup_visible_assets,
    _effective_available_from_round,
    _drop_conflicting_file_update_events,
    _drop_conflicting_subagent_file_effects,
    _flatten_subagent_state_effects,
    _inject_background_wait,
    _normalize_append_messages,
    _reconcile_asset_checks,
    _reconcile_file_checks,
    _reconcile_solution_command_asset_ids,
    _runtime_written_paths,
    _seed_source_files_for_read_checks,
    _session_channel_aliases,
    _session_channel_registry,
    _session_display_channel,
    _solution_touched_file_paths,
    _tag_asset_access_checks_diagnostic,
    assign_unresolved_channels,
    canonicalize_gold_channels,
    project_round_subagents,
    update_channel_aliases,
    update_channel_registry,
)


def _asset(asset_id: str, available_from_round: int = 1):
    return SimpleNamespace(id=asset_id, available_from_round=available_from_round)


def _asset_with_content(asset_id: str, available_from_round: int = 1, ocr_text: str = ""):
    """A bare asset that carries its own authored payload, not just an id."""
    return SimpleNamespace(
        id=asset_id, available_from_round=available_from_round, ocr_text=ocr_text
    )


def _assets_accessed_check(asset_id: str):
    return {
        "type": "effect",
        "effect_type": "assets_accessed",
        "condition": "field_contains",
        "expected": {"field": "id", "value": asset_id},
    }


# --------------------------------------------------------------------------- #
# G3: bare vs timestamped twin de-duplication                                 #
# --------------------------------------------------------------------------- #


def test_dedup_drops_bare_twin_keeps_timestamped():
    assets = [
        _asset("devops_dashboard_db_connections_chart"),
        _asset("devops_dashboard_db_connections_chart_20231027_101500"),
        _asset("wiki_network_docs_routing_topology_diagram"),
    ]
    kept = {a.id for a in _dedup_visible_assets(assets)}
    assert kept == {
        "devops_dashboard_db_connections_chart_20231027_101500",
        "wiki_network_docs_routing_topology_diagram",
    }


def test_dedup_preserves_distinct_timestamps_time_series():
    # A genuine time series (10:15 vs 10:18) must NOT be collapsed.
    assets = [
        _asset("db_connections_chart_20231027_101500"),
        _asset("db_connections_chart_20231027_101800"),
    ]
    kept = {a.id for a in _dedup_visible_assets(assets)}
    assert kept == {
        "db_connections_chart_20231027_101500",
        "db_connections_chart_20231027_101800",
    }


def test_dedup_keeps_lone_bare_asset():
    assets = [_asset("standalone_chart")]
    kept = {a.id for a in _dedup_visible_assets(assets)}
    assert kept == {"standalone_chart"}


def test_dedup_preserves_input_order():
    assets = [
        _asset("a_chart_20231027_101500"),
        _asset("a_chart"),
        _asset("b_diagram"),
    ]
    kept = [a.id for a in _dedup_visible_assets(assets)]
    assert kept == ["a_chart_20231027_101500", "b_diagram"]


# --------------------------------------------------------------------------- #
# G2: drop unsatisfiable assets_accessed checks                               #
# --------------------------------------------------------------------------- #


def test_reconcile_drops_check_for_invisible_asset():
    visible = [_asset("wiki_network_docs_routing_topology_diagram", 1)]
    checks = [
        _assets_accessed_check("devops_dashboard_db_connections_chart"),  # not visible
        _assets_accessed_check("wiki_network_docs_routing_topology_diagram"),
        {"type": "effect", "effect_type": "messages_sent", "expected": {}},
    ]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert dropped == ["devops_dashboard_db_connections_chart"]
    kinds = [c.get("effect_type") for c in reconciled]
    assert kinds == ["assets_accessed", "messages_sent"]


def test_reconcile_resolves_bare_check_against_timestamped_asset():
    # The de-dup keeps only the timestamped asset; a check written against the
    # bare id must still resolve (timestamp-agnostic) and be kept.
    visible = [_asset("devops_dashboard_db_connections_chart_20231027_101500", 2)]
    checks = [_assets_accessed_check("devops_dashboard_db_connections_chart")]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert dropped == []
    assert len(reconciled) == 1


def test_reconcile_keeps_non_asset_checks_untouched():
    visible = []
    checks = [
        {"type": "file_check", "path": "reports/x.md", "contains": "..."},
        {"type": "effect", "effect_type": "messages_sent", "expected": {}},
    ]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert dropped == []
    assert reconciled == checks


# --------------------------------------------------------------------------- #
# G1(b): instruction augmentation                                             #
# --------------------------------------------------------------------------- #


def test_augment_instruction_adds_guidance_when_asset_check_present():
    instr = "Analyze the database chart and resolve the conflict."
    out = _augment_instruction_for_asset_checks(
        instr, [_assets_accessed_check("some_chart")]
    )
    assert "asset list" in out
    assert "asset open" in out
    assert out.startswith(instr)


def test_augment_instruction_noop_without_asset_check():
    instr = "Send a message to the team."
    out = _augment_instruction_for_asset_checks(
        instr, [{"type": "effect", "effect_type": "messages_sent", "expected": {}}]
    )
    assert out == instr


def test_augment_instruction_is_idempotent():
    instr = "Analyze the chart."
    checks = [_assets_accessed_check("some_chart")]
    once = _augment_instruction_for_asset_checks(instr, checks)
    twice = _augment_instruction_for_asset_checks(once, checks)
    assert once == twice


# --------------------------------------------------------------------------- #
# base-key normalization parity with the effect checker                       #
# --------------------------------------------------------------------------- #


def test_asset_base_key_matches_effect_checker_semantics():
    from clasharness.evaluation.checkers.effect_checker import _normalize_asset_base

    for aid in (
        "devops_dashboard_db_connections_chart_20231027_101500",
        "devops_dashboard_db_connections_chart",
        "foo_r05",
        "api_gateway_chart",
        # Legacy double-prefix / path-like ids must fold identically in both.
        "workspace_workspace/flight_schedule",
        "workspace/flight_schedule_chart",
        "workspace_workspace/flight_schedule_20231027_105230",
    ):
        assert _asset_base_key(aid) == _normalize_asset_base(aid.lower())


def test_asset_base_key_folds_double_prefix_to_clean_id():
    # The phantom double-prefixed id and the clean, timestamped snapshot id the
    # generator now emits must collapse to the same base key so the reconcile
    # step and runtime grading treat them as one asset.
    clean = _asset_base_key("workspace_flight_schedule")
    assert _asset_base_key("workspace_workspace/flight_schedule") == clean
    assert (
        _asset_base_key("workspace_flight_schedule_20231027_105230") == clean
    )
    assert _asset_base_key("workspace/flight_schedule") == clean


def test_reconcile_canonicalizes_legacy_double_prefix_check():
    # A legacy check written against the doubled phantom id must resolve to the
    # clean visible snapshot id rather than being dropped as unsatisfiable.
    visible = [_asset("workspace_flight_schedule_20231027_105230", 2)]
    checks = [_assets_accessed_check("workspace_workspace/flight_schedule")]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert dropped == []
    assert reconciled[0]["expected"]["value"] == (
        "workspace_flight_schedule_20231027_105230"
    )


# --------------------------------------------------------------------------- #
# A1: effective availability hides untrusted bare ids until a real snapshot    #
# --------------------------------------------------------------------------- #


def test_effective_availability_bare_defers_to_earliest_timestamped_sibling():
    bare = _asset("workspace_flight_schedule", 1)  # untrusted r1
    ts_r2 = _asset("workspace_flight_schedule_20231027_105230", 2)
    ts_r4 = _asset("workspace_flight_schedule_20231027_180000", 4)
    all_assets = [bare, ts_r2, ts_r4]
    # The bare phantom is not "available" before its first real snapshot (r2).
    assert _effective_available_from_round(bare, all_assets) == 2
    # Timestamped snapshots keep their own declared round.
    assert _effective_available_from_round(ts_r2, all_assets) == 2
    assert _effective_available_from_round(ts_r4, all_assets) == 4


def test_effective_availability_lone_bare_asset_keeps_own_round():
    # A genuinely lone asset (e.g. an r1 login map with no timestamped twin) is
    # untouched, so it stays visible from its declared round.
    lone = _asset("workspace_customer_login_map", 1)
    assert _effective_available_from_round(lone, [lone]) == 1


def test_effective_availability_drops_early_round_phantom_check_end_to_end():
    # Simulate round r01 visibility for a scenario whose flight snapshot first
    # exists at r2: the phantom r1 bare id must NOT be visible, so the check
    # graded at r01 is dropped instead of impossible-to-satisfy.
    all_assets = [
        _asset("workspace_customer_login_map", 1),
        _asset("workspace_flight_schedule", 1),  # phantom bare, really r2+
        _asset("workspace_flight_schedule_20231027_105230", 2),
    ]
    round_index = 1
    visible = [
        a
        for a in all_assets
        if _effective_available_from_round(a, all_assets) <= round_index
    ]
    visible = _dedup_visible_assets(visible)
    visible_ids = {a.id for a in visible}
    assert visible_ids == {"workspace_customer_login_map"}

    checks = [
        _assets_accessed_check("workspace_customer_login_map"),
        _assets_accessed_check("workspace_flight_schedule"),
    ]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert dropped == ["workspace_flight_schedule"]
    kept_ids = [c["expected"]["value"] for c in reconciled]
    assert kept_ids == ["workspace_customer_login_map"]


# --------------------------------------------------------------------------- #
# A1b: deferral must fire ONLY on placeholders, never on a real visual         #
#                                                                              #
# Regression guard for the defect that produced every unwinnable round in the  #
# v8 and sonnet-5 corpora (27 rounds, 0/162 solver successes): a bare asset     #
# holding the round's answer was deferred past that round because an unrelated  #
# later snapshot collapsed onto its base key, so the value became unreadable    #
# while gold -- authored from resolved_value -- still passed.                   #
# --------------------------------------------------------------------------- #


def test_effective_availability_keeps_bare_asset_with_own_distinct_payload():
    # The bare board is authored content in its own right; the later snapshot is
    # a different reading of the same surface, not the thing this id stands in
    # for. It must stay visible from its declared round.
    bare = _asset_with_content(
        "ops_console_authorization_board_r2", 2, ocr_text="GATE-WRITE-ERR-4482"
    )
    later = _asset_with_content(
        "ops_console_authorization_board_r3_20240601_100200", 3, ocr_text="GATE OPEN"
    )
    assert _effective_available_from_round(bare, [bare, later]) == 2


def test_effective_availability_still_defers_bare_twin_duplicating_a_snapshot():
    # Same payload as its timestamped sibling => a genuine stand-in, so the
    # original deferral still applies. This is the case A1 was written for.
    same = "QUEUE DEPTH 12"
    bare = _asset_with_content("workspace_flight_schedule", 1, ocr_text=same)
    ts = _asset_with_content("workspace_flight_schedule_20231027_105230", 2, ocr_text=same)
    assert _effective_available_from_round(bare, [bare, ts]) == 2


def test_effective_availability_defers_contentless_bare_id():
    # No authored payload at all => nothing a solver could read => placeholder.
    bare = _asset_with_content("workspace_flight_schedule", 1, ocr_text="")
    ts = _asset_with_content(
        "workspace_flight_schedule_20231027_105230", 2, ocr_text="QUEUE DEPTH 12"
    )
    assert _effective_available_from_round(bare, [bare, ts]) == 2


def test_answer_bearing_bare_asset_is_visible_in_the_round_that_needs_it():
    # End-to-end shape of the shipped defect: r02 asks for a value that lives
    # only in the bare r2 board, while an r3 snapshot shares its base key.
    bare = _asset_with_content(
        "ops_console_authorization_board_r2", 2, ocr_text="GATE-WRITE-ERR-4482"
    )
    later = _asset_with_content(
        "ops_console_authorization_board_r3_20240601_100200", 3, ocr_text="GATE OPEN"
    )
    all_assets = [bare, later]
    visible = [
        a for a in all_assets if _effective_available_from_round(a, all_assets) <= 2
    ]
    visible = _dedup_visible_assets(visible)
    assert {a.id for a in visible} == {"ops_console_authorization_board_r2"}
    # ...and the value the round grades on is now readable from that surface.
    assert any("GATE-WRITE-ERR-4482" in a.ocr_text for a in visible)


def test_base_key_collapses_distinct_round_suffixes_documented():
    # Documents the collapse that made the defect possible: the _r<N> suffix,
    # which names the asset's own round, is stripped for grouping. Fixing the
    # deferral is what makes that collapse harmless -- this asserts the collapse
    # itself is still there, so the guard above is testing something real.
    assert _asset_base_key("ops_console_authorization_board_r2") == _asset_base_key(
        "ops_console_authorization_board_r3_20240601_100200"
    )


# --------------------------------------------------------------------------- #
# P1: canonical visible-asset map                                             #
# --------------------------------------------------------------------------- #


def test_canonical_map_prefers_latest_round_then_newest_timestamp():
    visible = [
        _asset("devops_dashboard_db_connections_chart_20231027_101500", 1),
        _asset("devops_dashboard_db_connections_chart_20231027_101800", 1),
        _asset("devops_dashboard_db_connections_chart_20231027_090000", 2),
    ]
    canonical = _canonical_visible_asset_map(visible)
    # Highest available_from_round (2) wins even though its timestamp is earlier.
    assert (
        canonical[_asset_base_key("devops_dashboard_db_connections_chart")]
        == "devops_dashboard_db_connections_chart_20231027_090000"
    )


def test_canonical_map_breaks_round_ties_by_greatest_id():
    visible = [
        _asset("flight_schedule_20231027_142201", 1),
        _asset("flight_schedule_20231027_142800", 1),
    ]
    canonical = _canonical_visible_asset_map(visible)
    assert (
        canonical[_asset_base_key("flight_schedule")]
        == "flight_schedule_20231027_142800"
    )


# --------------------------------------------------------------------------- #
# P1: rewrite surviving assets_accessed check ids to the canonical visible id #
# --------------------------------------------------------------------------- #


def test_reconcile_rewrites_bare_check_value_to_canonical_id():
    # The bare twin was pruned by de-dup; the surviving timestamped id must be
    # written into the check so the graded id names a real manifest entry.
    visible = [_asset("devops_dashboard_flight_schedule_20231027_142800", 4)]
    checks = [_assets_accessed_check("devops_dashboard_flight_schedule")]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert dropped == []
    assert reconciled[0]["expected"]["value"] == (
        "devops_dashboard_flight_schedule_20231027_142800"
    )


def test_reconcile_leaves_exact_match_check_value_untouched():
    visible = [_asset("devops_dashboard_risk_dashboard_status", 1)]
    checks = [_assets_accessed_check("devops_dashboard_risk_dashboard_status")]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert dropped == []
    assert reconciled[0]["expected"]["value"] == (
        "devops_dashboard_risk_dashboard_status"
    )


def test_reconcile_does_not_mutate_input_check_dict():
    visible = [_asset("db_connections_chart_20231027_101500", 1)]
    original = _assets_accessed_check("db_connections_chart")
    _reconcile_asset_checks([original], visible)
    # The shared input dict must be untouched (a copy is returned instead).
    assert original["expected"]["value"] == "db_connections_chart"


# --------------------------------------------------------------------------- #
# P1: point `asset ... --id X` solution commands at real manifest ids         #
# --------------------------------------------------------------------------- #


def test_reconcile_command_rewrites_bare_id_to_canonical():
    visible = [_asset("devops_dashboard_flight_schedule_20231027_142800", 4)]
    cmds = [
        "asset open --id devops_dashboard_flight_schedule",
        "file list",
        "DONE",
    ]
    out = _reconcile_solution_command_asset_ids(cmds, visible)
    assert out[0] == (
        "asset open --id devops_dashboard_flight_schedule_20231027_142800"
    )
    assert out[1] == "file list"
    assert out[2] == "DONE"


def test_reconcile_command_leaves_valid_id_untouched():
    visible = [_asset("devops_dashboard_risk_dashboard_status", 1)]
    cmds = ["asset open --id devops_dashboard_risk_dashboard_status"]
    assert _reconcile_solution_command_asset_ids(cmds, visible) == cmds


def test_reconcile_command_ignores_non_asset_commands():
    visible = [_asset("db_connections_chart_20231027_101500", 1)]
    cmds = ["file read --path config.json", "message send --channel ops --message hi"]
    assert _reconcile_solution_command_asset_ids(cmds, visible) == cmds


def test_reconcile_command_covers_info_and_read_text_subcommands():
    visible = [_asset("wiki_topology_diagram_20231027_101500", 1)]
    cmds = [
        "asset info --id wiki_topology_diagram",
        "asset read-text --id wiki_topology_diagram",
    ]
    out = _reconcile_solution_command_asset_ids(cmds, visible)
    assert out[0] == "asset info --id wiki_topology_diagram_20231027_101500"
    assert out[1] == "asset read-text --id wiki_topology_diagram_20231027_101500"


# --------------------------------------------------------------------------- #
# P1b: drop pure-read commands whose asset is not visible this round (M14      #
# residue: bare/phantom twin hidden by effective-availability until later)     #
# --------------------------------------------------------------------------- #


def test_reconcile_command_drops_dangling_invisible_asset():
    # `ops_console_devops_dashboard` names no asset visible this round (not even
    # by base key), so the pure-read command is a guaranteed no-op and must be
    # dropped, while the visible read and non-asset commands are preserved.
    visible = [_asset("status_page_alert_20231024_101502", 1)]
    cmds = [
        "asset open --id status_page_alert_20231024_101502",  # visible -> keep
        "asset read-text --id ops_console_devops_dashboard",  # invisible -> drop
        "message send --channel ops --message hi",            # non-asset -> keep
        "DONE",
    ]
    out = _reconcile_solution_command_asset_ids(cmds, visible)
    assert out == [
        "asset open --id status_page_alert_20231024_101502",
        "message send --channel ops --message hi",
        "DONE",
    ]


def test_reconcile_command_drops_all_asset_subcommand_kinds_when_invisible():
    visible = [_asset("some_visible_chart_20231024_101502", 1)]
    for sub in ("open", "info", "read-text", "describe"):
        cmds = [f"asset {sub} --id ghost_asset_not_present", "DONE"]
        out = _reconcile_solution_command_asset_ids(cmds, visible)
        assert out == ["DONE"], f"{sub} dangling read not dropped: {out}"


def test_reconcile_command_guard_restores_originals_when_all_dangling():
    # Pathological: every command is a dangling read. Rather than emit an empty
    # trajectory, the originals are restored.
    visible = [_asset("unrelated_chart_20231024_101502", 1)]
    cmds = [
        "asset open --id ghost_a",
        "asset read-text --id ghost_b",
    ]
    assert _reconcile_solution_command_asset_ids(cmds, visible) == cmds


# --------------------------------------------------------------------------- #
# P1b (check side): a `command_check` grading a dropped dangling `asset ...`   #
# read must be dropped/rewritten in lock-step with the trajectory command, or  #
# the compiled task fails local simulation (the command is never issued).      #
# --------------------------------------------------------------------------- #


def test_reconcile_asset_checks_drops_orphaned_command_check_for_dangling_read():
    # When P1b drops `asset open --id ops_console_post_incident_metrics_...` from
    # the trajectory, its paired command_check can never pass and must go too. The
    # visible read and the non-asset command_check are preserved.
    visible = [_asset("status_page_alert_20231024_101502", 1)]
    checks = [
        {"type": "command_check",
         "command": "asset open --id status_page_alert_20231024_101502"},   # visible -> keep
        {"type": "command_check",
         "command": "asset open --id ops_console_post_incident_metrics_20231024_104100"},  # dangling -> drop
        {"type": "command_check",
         "command": "message send --channel ops --message hi"},             # non-asset -> keep
    ]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert [c["command"] for c in reconciled] == [
        "asset open --id status_page_alert_20231024_101502",
        "message send --channel ops --message hi",
    ]
    assert any("ops_console_post_incident_metrics" in d for d in dropped)


def test_reconcile_asset_checks_rewrites_command_check_bare_twin_to_canonical():
    # A command_check written against a pruned bare twin is pointed at the
    # canonical visible id (mirrors the P1 rewrite of the solution command).
    visible = [_asset("wiki_topology_diagram_20231027_101500", 1)]
    checks = [{"type": "command_check", "command": "asset open --id wiki_topology_diagram"}]
    reconciled, dropped = _reconcile_asset_checks(checks, visible)
    assert reconciled[0]["command"] == (
        "asset open --id wiki_topology_diagram_20231027_101500"
    )
    assert dropped == []


def test_command_check_and_solution_command_drop_in_lockstep():
    # The check side and the command side must make the SAME decision for the same
    # dangling id, so a reconciled trajectory and its checks stay consistent.
    visible = [_asset("visible_chart_20231024_101502", 1)]
    dangling = "asset open --id ghost_asset_20231024_999999"
    assert _reconcile_solution_command_asset_ids([dangling, "DONE"], visible) == ["DONE"]
    reconciled, dropped = _reconcile_asset_checks(
        [{"type": "command_check", "command": dangling}], visible
    )
    assert reconciled == []
    assert len(dropped) == 1


# --------------------------------------------------------------------------- #
# P1: drop file_check contains/equals the expert's own edit contradicts       #
# --------------------------------------------------------------------------- #


def test_reconcile_file_checks_drops_contradicted_contains():
    sol = [
        'file create --path config.json --content "endpoint: prod\nkey: /keys/GOOD"',
        "DONE",
    ]
    checks = [
        {"type": "file_check", "path": "config.json", "condition": "contains",
         "expected": "/keys/GOOD"},        # satisfied -> keep
        {"type": "file_check", "path": "config.json", "condition": "contains",
         "expected": "/keys/OLD_BROKEN"},  # contradicted by the rewrite -> drop
    ]
    reconciled, dropped = _reconcile_file_checks(checks, sol)
    assert len(reconciled) == 1
    assert reconciled[0]["expected"] == "/keys/GOOD"
    assert len(dropped) == 1
    assert "/keys/OLD_BROKEN" in dropped[0]


def test_reconcile_file_checks_supports_contains_shorthand_key():
    sol = ['file create --path config.json --content "value: new"']
    checks = [{"type": "file_check", "path": "config.json", "contains": "value: old"}]
    reconciled, dropped = _reconcile_file_checks(checks, sol)
    assert reconciled == []
    assert len(dropped) == 1


def test_reconcile_file_checks_drops_equals_mismatch():
    sol = ['file create --path status.txt --content "READY"']
    checks = [{"type": "file_check", "path": "status.txt", "condition": "equals",
               "expected": "PENDING"}]
    reconciled, dropped = _reconcile_file_checks(checks, sol)
    assert reconciled == []
    assert len(dropped) == 1


def test_reconcile_file_checks_keeps_files_not_written_by_expert():
    # runtime.log is produced by a subagent, not the expert trajectory, so its
    # check must be preserved even though the expert writes a different file.
    sol = ['file create --path config.json --content "ok"']
    checks = [
        {"type": "file_check", "path": "runtime.log", "condition": "contains",
         "expected": "started"},
    ]
    reconciled, dropped = _reconcile_file_checks(checks, sol)
    assert reconciled == checks
    assert dropped == []


def test_reconcile_file_checks_never_drops_non_content_conditions():
    sol = ['file create --path a.txt --content "x"']
    checks = [
        {"type": "file_check", "path": "a.txt", "condition": "exists"},
        {"type": "file_check", "path": "a.txt", "condition": "not_exists"},
    ]
    reconciled, dropped = _reconcile_file_checks(checks, sol)
    assert reconciled == checks
    assert dropped == []


def test_reconcile_file_checks_normalizes_workspace_path_prefix():
    sol = ['file create --path workspace/config.json --content "final"']
    checks = [{"type": "file_check", "path": "/config.json", "condition": "contains",
               "expected": "stale"}]
    reconciled, dropped = _reconcile_file_checks(checks, sol)
    assert reconciled == []
    assert len(dropped) == 1


def test_reconcile_file_checks_noop_without_file_creates():
    checks = [{"type": "file_check", "path": "config.json", "contains": "x"}]
    reconciled, dropped = _reconcile_file_checks(checks, ["file list", "DONE"])
    assert reconciled == checks
    assert dropped == []


# --------------------------------------------------------------------------- #
# B3: flatten nested subagent state_effects to runtime-applicable forms        #
# --------------------------------------------------------------------------- #


def test_flatten_unwraps_nested_effect_type_file_and_command():
    raw = [
        {
            "subagent_name": "system_admin",
            "trigger_conditions": {"message_contains": ["restart"]},
            "state_effects": [
                {"effect_type": "file_write", "path": "config/x.json",
                 "content": "fixed"},
                {"effect_type": "command_execution",
                 "command": "system gateway restart"},
            ],
        }
    ]
    out = _flatten_subagent_state_effects(raw)
    assert {"type": "file_create", "path": "config/x.json", "content": "fixed"} in out
    assert "system gateway restart" in out


def test_flatten_keeps_plain_cli_strings():
    assert _flatten_subagent_state_effects(["run tests", "  ", "deploy"]) == [
        "run tests",
        "deploy",
    ]


def test_flatten_serializes_non_string_file_content_to_json():
    out = _flatten_subagent_state_effects(
        [{"type": "file_create", "path": "a.json", "content": {"k": 1}}]
    )
    assert out == [{"type": "file_create", "path": "a.json", "content": '{"k": 1}'}]


def test_flatten_dedupes_repeated_effect_variants():
    raw = [
        {"effect_type": "command_execution", "command": "system restart"},
        {"effect_type": "command_execution", "command": "system restart"},
    ]
    assert _flatten_subagent_state_effects(raw) == ["system restart"]


def test_flatten_handles_commands_executed_and_files_written_keys():
    raw = [{
        "commands_executed": ["svc reload"],
        "files_written": [{"path": "out.txt", "content": "done"}],
    }]
    out = _flatten_subagent_state_effects(raw)
    assert "svc reload" in out
    assert {"type": "file_create", "path": "out.txt", "content": "done"} in out


# --------------------------------------------------------------------------- #
# B3: expert-authored path detection + collision filtering                     #
# --------------------------------------------------------------------------- #


def test_solution_touched_paths_covers_write_verbs():
    sol = [
        'file create --path a.txt --content "x"',
        'file update --path workspace/b.json --content "y"',
        "file list",
    ]
    assert _solution_touched_file_paths(sol) == {"a.txt", "b.json"}


def test_drop_subagent_effect_that_clobbers_expert_file():
    effects = [
        {"type": "file_create", "path": "config.json", "content": "subagent"},
        {"type": "file_create", "path": "other.txt", "content": "keep"},
        "run tests",
    ]
    kept = _drop_conflicting_subagent_file_effects(effects, {"config.json"})
    assert {"type": "file_create", "path": "other.txt", "content": "keep"} in kept
    assert "run tests" in kept
    assert all(e.get("path") != "config.json" for e in kept if isinstance(e, dict))


def test_drop_subagent_effect_noop_without_expert_files():
    effects = [{"type": "file_create", "path": "config.json", "content": "s"}]
    assert _drop_conflicting_subagent_file_effects(effects, set()) == effects


def test_drop_file_update_event_that_clobbers_expert_file():
    events = [
        {"type": "file_update", "path": "config.json", "new_content": "conflict"},
        {"type": "file_update", "path": "untouched.txt", "new_content": "keep"},
        {"type": "session_append", "message": {"text": "hi"}},
    ]
    kept = _drop_conflicting_file_update_events(events, {"config.json"})
    paths = [e.get("path") for e in kept if e.get("type") == "file_update"]
    assert paths == ["untouched.txt"]
    assert any(e.get("type") == "session_append" for e in kept)


def test_drop_file_update_normalizes_workspace_prefix():
    events = [{"type": "file_update", "path": "workspace/config.json",
               "new_content": "conflict"}]
    assert _drop_conflicting_file_update_events(events, {"config.json"}) == []


def test_drop_file_update_noop_without_expert_files():
    events = [{"type": "file_update", "path": "config.json", "new_content": "c"}]
    assert _drop_conflicting_file_update_events(events, set()) == events


# --------------------------------------------------------------------------- #
# B3: inject a wait so time-triggered background events fire before grading     #
# --------------------------------------------------------------------------- #


def test_inject_wait_before_done_marker():
    sol = ['file create --path a.txt --content "x"', "DONE"]
    events = [{"type": "subagent_completion", "trigger_time": 30}]
    out = _inject_background_wait(sol, events)
    assert out == ['file create --path a.txt --content "x"', "wait 30", "DONE"]


def test_inject_wait_appends_when_no_terminal_marker():
    sol = ["file list"]
    events = [{"type": "file_update", "trigger_time": 15}]
    assert _inject_background_wait(sol, events) == ["file list", "wait 15"]


def test_inject_wait_noop_when_no_positive_trigger():
    sol = ["file list", "DONE"]
    events = [{"type": "session_append", "trigger_time": 0}]
    assert _inject_background_wait(sol, events) == sol


def test_inject_wait_respects_existing_longer_wait():
    sol = ["wait 99", "DONE"]
    events = [{"type": "subagent_completion", "trigger_time": 30}]
    assert _inject_background_wait(sol, events) == sol


def test_inject_wait_upgrades_short_existing_wait():
    sol = ["wait 5", "DONE"]
    events = [{"type": "subagent_completion", "trigger_time": 30}]
    out = _inject_background_wait(sol, events)
    assert "wait 30" in out


# --------------------------------------------------------------------------- #
# P2: seed read-evidence values into the source file the check reads            #
# --------------------------------------------------------------------------- #


def test_runtime_written_paths_covers_updates_and_subagent_effects():
    bg = [
        {"type": "file_update", "path": "workspace/config.yaml", "new_content": "x"},
        {
            "type": "subagent_completion",
            "state_effects": [
                {"type": "file_create", "path": "report.md", "content": "z"},
                'file create --path notes/from_sub.txt --content "zzz"',
                "run tests",
            ],
        },
    ]
    assert _runtime_written_paths(bg) == {
        "config.yaml",
        "report.md",
        "notes/from_sub.txt",
    }


def test_runtime_written_paths_empty_without_file_effects():
    bg = [{"type": "session_append", "message": {"text": "hi"}}]
    assert _runtime_written_paths(bg) == set()


def test_seed_source_file_injects_missing_read_value():
    # A read-evidence check whose value the LLM left out of the seeded file:
    # amend the pre-existing source file so replaying the trajectory can pass it.
    overrides = {"workspace/evidence.txt": "baseline content\n"}
    checks = [{"type": "file_check", "path": "evidence.txt", "contains": "NODE_9921"}]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert len(seeded) == 1
    assert "NODE_9921" in overrides["workspace/evidence.txt"]


def test_seed_source_file_supports_condition_expected_form():
    overrides = {"workspace/e.txt": "base"}
    checks = [
        {"type": "file_check", "path": "e.txt", "condition": "contains",
         "expected": "ZZZ"},
    ]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert len(seeded) == 1 and "ZZZ" in overrides["workspace/e.txt"]


def test_seed_source_file_noop_when_value_already_present():
    overrides = {"workspace/evidence.txt": "already has NODE_9921 here\n"}
    checks = [{"type": "file_check", "path": "evidence.txt", "contains": "NODE_9921"}]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert seeded == []
    assert overrides["workspace/evidence.txt"] == "already has NODE_9921 here\n"


def test_seed_source_file_skips_reserved_paths():
    # A file the expert / subagent / background update writes is graded by the
    # write, not the seed: never pre-fill it.
    overrides = {"workspace/out.json": "{}"}
    checks = [{"type": "file_check", "path": "out.json", "contains": "VAL"}]
    seeded = _seed_source_files_for_read_checks(checks, overrides, {"out.json"})
    assert seeded == []
    assert overrides["workspace/out.json"] == "{}"


def test_seed_source_file_never_fabricates_missing_file():
    overrides = {"workspace/evidence.txt": "x"}
    checks = [{"type": "file_check", "path": "ghost.txt", "contains": "VAL"}]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert seeded == []
    assert "workspace/ghost.txt" not in overrides


def test_seed_source_file_skips_code_files():
    # A decisive business value never belongs textually inside a script.
    overrides = {"workspace/scripts/remap.py": "print('hi')\n"}
    checks = [
        {"type": "file_check", "path": "scripts/remap.py", "contains": "MAP_NODE_9921"},
    ]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert seeded == []
    assert overrides["workspace/scripts/remap.py"] == "print('hi')\n"


def test_seed_source_file_normalizes_check_path_prefix():
    overrides = {"workspace/evidence.txt": "base"}
    checks = [{"type": "file_check", "path": "/evidence.txt", "contains": "QQQ"}]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert len(seeded) == 1 and "QQQ" in overrides["workspace/evidence.txt"]


def test_seed_source_file_handles_multiple_values_same_file():
    overrides = {"workspace/e.txt": "base"}
    checks = [
        {"type": "file_check", "path": "e.txt", "contains": "AAA"},
        {"type": "file_check", "path": "e.txt", "contains": "BBB"},
    ]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert len(seeded) == 2
    assert "AAA" in overrides["workspace/e.txt"]
    assert "BBB" in overrides["workspace/e.txt"]


def test_seed_source_file_ignores_non_file_and_non_contains_checks():
    overrides = {"workspace/e.txt": "base"}
    checks = [
        {"type": "file_check", "path": "e.txt", "condition": "exists"},
        {"type": "effect", "effect_type": "messages_sent"},
    ]
    seeded = _seed_source_files_for_read_checks(checks, overrides, set())
    assert seeded == []
    assert overrides["workspace/e.txt"] == "base"


# --------------------------------------------------------------------------- #
# session_appends schema-drift normalization                                  #
#                                                                             #
# _compile_round_to_task and the spec materializer read every appended chat   #
# message via ``msg.get(...)``. The generator emits two shapes: the canonical #
# ``{"messages": [ {..dict..} ]}`` and a flat singular                        #
# ``{"sender","timestamp","message": <str>}``. A bare-string ``message`` used #
# to be wrapped as ``[<str>]`` and then hit ``str.get`` -> AttributeError     #
# ('str' object has no attribute 'get'), crashing generation for the whole    #
# version. ``_normalize_append_messages`` coerces every entry to a dict.      #
# --------------------------------------------------------------------------- #


def test_normalize_append_messages_canonical_dicts_passthrough():
    app = {
        "session_id": "s1",
        "messages": [
            {"message_id": "m1", "sender": "a", "text": "hello"},
            {"message_id": "m2", "sender": "b", "text": "world"},
        ],
    }
    out = _normalize_append_messages(app)
    assert out == app["messages"]
    # Same dict objects passed through unchanged (no copying/mutation).
    assert out[0] is app["messages"][0]


def test_normalize_append_messages_flat_string_coerced_and_hoists_metadata():
    # The exact payment_chargeback_dispute_v01 shape that crashed generation.
    app = {
        "session_id": "session_01_fraud_escalation",
        "sender": "risk_agent",
        "timestamp": "2023-10-24T08:15:00Z",
        "message": "The customer was on flight LH-2401 during the layover window.",
    }
    out = _normalize_append_messages(app)
    assert len(out) == 1
    msg = out[0]
    # Coerced to a dict so downstream msg.get(...) is safe.
    assert isinstance(msg, dict)
    assert msg["text"] == app["message"]
    # app-level author/timestamp are hoisted into the message (not lost).
    assert msg["sender"] == "risk_agent"
    assert msg["timestamp"] == "2023-10-24T08:15:00Z"
    assert msg["message_id"] == ""
    # Downstream consumers only ever call .get(...) on the result - prove it.
    assert msg.get("text") == app["message"]
    assert msg.get("attachments", []) == []


def test_normalize_append_messages_singular_dict_message():
    # Flat shape where `message` is already a dict is passed through as-is.
    inner = {"message_id": "m9", "sender": "x", "text": "hi"}
    app = {"session_id": "s1", "message": inner}
    out = _normalize_append_messages(app)
    assert out == [inner]
    assert out[0] is inner


def test_normalize_append_messages_missing_metadata_defaults_to_empty():
    app = {"session_id": "s1", "message": "bare text only"}
    out = _normalize_append_messages(app)
    assert out == [
        {"message_id": "", "timestamp": "", "sender": "", "text": "bare text only"}
    ]


def test_normalize_append_messages_skips_non_str_non_dict_junk():
    app = {"messages": [{"text": "keep"}, 123, None, ["nested"], "coerce-me"]}
    out = _normalize_append_messages(app)
    assert out == [
        {"text": "keep"},
        {"message_id": "", "timestamp": "", "sender": "", "text": "coerce-me"},
    ]


def test_normalize_append_messages_empty_when_no_messages():
    assert _normalize_append_messages({"session_id": "s1"}) == []
    assert _normalize_append_messages({"session_id": "s1", "messages": []}) == []


def test_normalize_append_messages_prefers_messages_over_singular():
    # When both keys exist, the canonical list wins (non-empty messages).
    app = {
        "messages": [{"text": "canonical"}],
        "message": "ignored singular",
    }
    out = _normalize_append_messages(app)
    assert out == [{"text": "canonical"}]


def test_normalize_append_messages_mixed_list_coerces_strings_in_place():
    # A messages list that mixes dicts and bare strings (partial drift).
    app = {
        "sender": "bot",
        "timestamp": "t0",
        "messages": [{"text": "a"}, "b"],
    }
    out = _normalize_append_messages(app)
    assert out[0] == {"text": "a"}
    assert out[1] == {
        "message_id": "",
        "timestamp": "t0",
        "sender": "bot",
        "text": "b",
    }


# --------------------------------------------------------------------------- #
# M7 Axis A: decisive-asset access-check promotion (vision grading)            #
# --------------------------------------------------------------------------- #


def test_tag_asset_checks_legacy_all_diagnostic_when_no_decisive():
    # Non-opted seeds (decisive_asset_id=None) keep every assets_accessed check
    # diagnostic -- byte-identical legacy behaviour (E6).
    checks = [
        _assets_accessed_check("receipt_photo"),
        _assets_accessed_check("conversion_rate_chart"),
    ]
    out = _tag_asset_access_checks_diagnostic(checks)
    assert [c["role"] for c in out] == ["diagnostic", "diagnostic"]


def test_tag_asset_checks_promotes_only_decisive_to_required():
    # When the round is verifiably vision-confined, exactly the decisive asset's
    # access check is promoted to required; corroborating stays diagnostic.
    checks = [
        _assets_accessed_check("receipt_photo"),
        _assets_accessed_check("conversion_rate_chart"),
    ]
    out = _tag_asset_access_checks_diagnostic(
        checks, decisive_asset_id="receipt_photo"
    )
    by_val = {
        (c.get("expected") or {}).get("value"): c["role"] for c in out
    }
    assert by_val["receipt_photo"] == "required"
    assert by_val["conversion_rate_chart"] == "diagnostic"


def test_tag_asset_checks_unknown_decisive_leaves_all_diagnostic():
    # A decisive id that matches no check must not promote anything (fail-safe).
    checks = [_assets_accessed_check("receipt_photo")]
    out = _tag_asset_access_checks_diagnostic(
        checks, decisive_asset_id="does_not_exist"
    )
    assert out[0]["role"] == "diagnostic"


def test_tag_asset_checks_ignores_non_asset_checks():
    # Only assets_accessed effect checks are touched; other checks pass through.
    other = {"type": "effect", "effect_type": "messages_sent"}
    checks = [_assets_accessed_check("receipt_photo"), dict(other)]
    out = _tag_asset_access_checks_diagnostic(
        checks, decisive_asset_id="receipt_photo"
    )
    assert out[0]["role"] == "required"
    assert "role" not in out[1]  # untouched


# --------------------------------------------------------------------------- #
# Channel vocabulary: one name for one conversation                            #
#                                                                              #
# The graded channel identity is a session's DISPLAY name, derived HERE, but   #
# the gold trajectory is authored one stage earlier where only the raw         #
# session_id is visible. Measured on the v6 corpus that split produced 54 gold #
# --channel values naming a channel that does not exist and 37 grading         #
# criteria quoting those invented names (against 0 quoting a real one).        #
# --------------------------------------------------------------------------- #


def _session(session_id: str, channel: str = "default"):
    return {"session_id": session_id, "channel": channel}


def test_display_channel_reads_the_raw_layer1b_dict():
    # The generator holds dicts, the materializer holds objects; one accessor
    # must serve both or the two stages drift onto different registries.
    assert _session_display_channel(_session("channel_ops_triage")) == "#ops-triage"
    assert (
        _session_display_channel(SimpleNamespace(session_id="channel_ops_triage",
                                                 channel="default"))
        == "#ops-triage"
    )


def test_display_channel_prefers_an_explicit_channel_over_the_storage_key():
    assert (
        _session_display_channel(_session("channel_ops_triage", "#war-room"))
        == "#war-room"
    )


def test_display_channel_does_not_double_prefix_a_hashed_session_id():
    # Regression: '#incident-ops' as a session_id became '##incident-ops', a
    # name no participant ever utters, and was then stamped onto every
    # reconciled sub-agent reply. Seen twice in the v5 corpus.
    assert _session_display_channel(_session("#incident-ops")) == "#incident-ops"


def test_display_channel_empty_when_the_session_names_nothing():
    assert _session_display_channel(_session("")) == ""


def test_aliases_admit_only_provable_forms():
    aliases = _session_channel_aliases([_session("channel_ops_triage")])
    # display name, display name without '#', and the storage key it came from
    assert aliases["#ops-triage"] == "#ops-triage"
    assert aliases["ops-triage"] == "#ops-triage"
    assert aliases["channel_ops_triage"] == "#ops-triage"
    # anything else is a genuine authoring choice, not a spelling of this one
    assert "ops" not in aliases
    assert "triage" not in aliases


def test_aliases_empty_without_sessions():
    # A scenario really can ship none (one v6 scenario does). Callers rely on
    # "no vocabulary" meaning "enforce nothing", so this must not invent one.
    assert _session_channel_aliases([]) == {}


def test_canonicalize_rewrites_provable_aliases():
    aliases = _session_channel_aliases([_session("channel_ops_triage")])
    out, unresolved = canonicalize_gold_channels(
        [
            "message send --channel 'ops-triage' --target 'A' --message 'x'",
            "message send --channel 'channel_ops_triage' --target 'B' --message 'y'",
            "message send --channel '#ops-triage' --target 'C' --message 'z'",
        ],
        aliases,
    )
    assert unresolved == []
    assert all("--channel '#ops-triage'" in c for c in out)


def test_canonicalize_reports_rather_than_guesses():
    # The decisive property. A similarity score maps 'prod-cutover-war-room'
    # onto '#war-room' and silently merges two channels the scenario keeps
    # apart; the gold trajectory is not the place for that bet.
    aliases = _session_channel_aliases(
        [_session("channel_war_room"), _session("channel_ops_triage")]
    )
    out, unresolved = canonicalize_gold_channels(
        ["message send --channel '#prod-cutover-war-room' --target 'A' --message 'x'"],
        aliases,
    )
    assert unresolved == ["#prod-cutover-war-room"]
    assert out[0].count("--channel '#prod-cutover-war-room'") == 1  # untouched


def test_canonicalize_covers_reads_not_just_utterances():
    # 'message search' emits no graded effect, so it is out of scope for
    # _is_message_send_effect -- but a search of a channel that does not exist
    # returns nothing, leaving the gold's next step resting on evidence it
    # never actually read. Reachability, not audibility, is the test here.
    aliases = _session_channel_aliases([_session("channel_ops_triage")])
    out, unresolved = canonicalize_gold_channels(
        [
            "message search --channel 'ops-triage' --query 'scope'",
            "message search --channel '#risk-fraud-ops' --query 'scope'",
        ],
        aliases,
    )
    assert "--channel '#ops-triage'" in out[0]
    assert unresolved == ["#risk-fraud-ops"]


def test_canonicalize_leaves_commands_that_address_no_channel():
    aliases = _session_channel_aliases([_session("channel_ops_triage")])
    cmds = [
        "file read --path 'workspace/notes.md'",
        "asset open --id 'dashboard'",
        "DONE",
    ]
    out, unresolved = canonicalize_gold_channels(list(cmds), aliases)
    assert out == cmds
    assert unresolved == []


def test_canonicalize_is_a_noop_without_a_vocabulary():
    cmds = ["message send --channel '#anything' --target 'A' --message 'x'"]
    out, unresolved = canonicalize_gold_channels(list(cmds), {})
    assert out == cmds
    assert unresolved == []


def test_assignment_keeps_distinct_invented_names_distinct():
    # The last-attempt escape's one real guarantee: a round that ran two
    # conversations still runs two. Measured on v6, the invented names share no
    # tokens with the real ones, so a similarity score would collapse both.
    sessions = [_session("channel_proj_migration_sync"), _session("channel_ops_escalations")]
    aliases = _session_channel_aliases(sessions)
    registry = _session_channel_registry(sessions)
    out, mapping = assign_unresolved_channels(
        [
            "message send --channel 'prod-access-permits' --target 'A' --message 'x'",
            "message send --channel 'prod-release-ops' --target 'B' --message 'y'",
            "message send --channel 'prod-access-permits' --target 'C' --message 'z'",
        ],
        aliases,
        registry,
    )
    landed = [c.split("--channel '")[1].split("'")[0] for c in out]
    assert landed[0] == landed[2]           # one invented name, one destination
    assert landed[0] != landed[1]           # two invented names stay apart
    assert set(landed) <= set(registry)     # and both are channels that exist
    assert mapping == {
        "prod-access-permits": "#proj-migration-sync",
        "prod-release-ops": "#ops-escalations",
    }


def test_assignment_never_touches_a_channel_that_was_already_right():
    sessions = [_session("channel_ops_triage")]
    cmd = "message send --channel '#ops-triage' --target 'A' --message 'x'"
    out, mapping = assign_unresolved_channels(
        [cmd],
        _session_channel_aliases(sessions),
        _session_channel_registry(sessions),
    )
    assert out == [cmd]
    assert mapping == {}


def test_assignment_is_a_noop_without_a_vocabulary():
    cmds = ["message send --channel '#anything' --target 'A' --message 'x'"]
    out, mapping = assign_unresolved_channels(list(cmds), {}, [])
    assert out == cmds
    assert mapping == {}


# --------------------------------------------------------------------------- #
# Projecting sub-agents for the generation-time gate                           #
# --------------------------------------------------------------------------- #


def test_projection_supplies_the_reply_channel_the_author_never_writes():
    # THE regression. An authored mock has no 'reply_channel' -- it is filled in
    # at materialization -- so a gate reading the raw mock rebuilt every reply
    # without its 'in #channel' and graded a transcript the runtime never
    # produces: 0/18 line sets matched on v5 and v6 alike.
    authored = {"agent_role": "Quinn_X", "mock_response": "scope check failed"}
    out = project_round_subagents(
        [authored],
        [_session("channel_prod_access_requests")],
        ["message send --channel '#prod-access-requests' --target 'Quinn_X' --message 'status?'"],
    )
    assert out[0]["reply_channel"] == "#prod-access-requests"
    assert "reply_channel" not in authored  # caller's artifact left alone


def test_projection_reconciles_channels_inside_state_effects():
    out = project_round_subagents(
        [{
            "agent_role": "Quinn_X",
            "mock_response": "ack",
            "state_effects": [
                "message send --channel '#prod access requests' --target 'A' --message 'done'"
            ],
        }],
        [_session("channel_prod_access_requests")],
        [],
    )
    assert "#prod-access-requests" in out[0]["state_effects"][0]


def test_projection_falls_back_to_the_primary_channel():
    # A sub-agent whose own effects name no channel still has to reply
    # somewhere; the runtime picks the round's primary channel, so must this.
    out = project_round_subagents(
        [{"agent_role": "Quinn_X", "mock_response": "ack"}],
        [_session("channel_ops_triage"), _session("channel_side_bar")],
        ["message send --channel '#ops-triage' --target 'Quinn_X' --message 'go'"],
    )
    assert out[0]["reply_channel"] == "#ops-triage"


def test_projection_survives_a_scenario_with_no_sessions():
    # coordinated_containment_v02 ships none. Must not raise.
    out = project_round_subagents(
        [{"agent_role": "Quinn_X", "mock_response": "ack"}], [], []
    )
    assert out[0]["reply_channel"] == ""


def test_projection_skips_junk_entries():
    out = project_round_subagents(
        ["not a dict", {"agent_role": "Q", "mock_response": "ack"}],
        [_session("channel_ops_triage")],
        [],
    )
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Channel vocabulary: the fallback witness
# ---------------------------------------------------------------------------


def _upd(*channels):
    return {"session_appends": [{"channel_or_thread": c} for c in channels]}


class TestUpdateChannelRegistry:
    """Where the vocabulary comes from when the sessions declare nothing.

    ``coordinated_containment_v02`` ships one degenerate session
    (``session_id: ""``, ``channel: "default"``) so
    ``_session_channel_registry`` returns nothing for it -- yet its updates post
    into two channels every round and its gold names them. This is the second
    witness that keeps the gate from being a no-op there.
    """

    def test_it_collects_names_in_first_appearance_order(self):
        assert update_channel_registry([
            _upd("#war-room", "#secops"),
            _upd("#war-room"),
        ]) == ["#war-room", "#secops"]

    def test_a_bare_name_is_prefixed_like_a_session_display_name(self):
        assert update_channel_registry([_upd("secops")]) == ["#secops"]

    def test_the_literal_default_is_not_a_channel(self):
        assert update_channel_registry([_upd("default", "#real")]) == ["#real"]

    def test_an_already_hashed_name_is_not_double_prefixed(self):
        # The '##incident-ops' defect, in the other derivation path.
        assert update_channel_registry([_upd("#ops")]) == ["#ops"]

    def test_empty_and_missing_shapes_are_survivable(self):
        assert update_channel_registry([]) == []
        assert update_channel_registry([{}]) == []
        assert update_channel_registry([_upd("", "   ")]) == []

    def test_aliases_admit_the_name_with_and_without_the_hash(self):
        aliases = update_channel_aliases([_upd("#war-room")])
        assert aliases["#war-room"] == "#war-room"
        assert aliases["war-room"] == "#war-room"

    def test_aliases_are_casefolded_like_the_session_ones(self):
        assert update_channel_aliases([_upd("#War-Room")])["#war-room"] == (
            "#War-Room"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
