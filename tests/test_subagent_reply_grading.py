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

"""Option C: mock subagents actually speak and are graded.

Root cause these tests lock in:
  - Runtime (R1/R2): a mock subagent's preset ``mock_response`` (and any
    message-send ``state_effect``) is routed into ``effects["messages_sent"]`` -
    the sink the grader reads - instead of only ``command_history`` / the backend
    message pool (which ``merge_backend_state_into_eval`` never backfills).
  - Generation (G1): the materializer reconciles the LLM-invented channel of a
    subagent's effect (e.g. ``#ops-triage``) onto the canonical shipped DISPLAY
    name (``#operations-triage``) - the identity the graded criteria and the
    solver's ``message search`` both speak, as opposed to the ``session_id``
    storage key - and records a ``reply_channel`` so the runtime can route the
    reply where the grader/solver can see it.
  - Grading: ``EffectChecker`` surfaces the message ``sender`` so a criterion
    phrased about a specific speaker is gradeable by attribution.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.core.environment import CLASHarnessEnv


# A checked-in copy of one ``<slug>_spec.json`` from the v2 corpus. This used
# to read the generated corpus in-tree
# (``clasharness/data/scenarios/conflict_vlm_v2_3.6_flash``), which made the
# spec-backed test below pass only on a machine that still had that ~20 GB
# output directory. Generated corpora are disposable, so the one spec the test
# needs is vendored here instead; only the spec is required (no assets are
# opened). Refresh it by re-copying the file if the spec schema changes.
#
# ``.resolve()`` is deliberately omitted: under sandboxed build runners
# ``__file__`` can be a symlink into a content-addressed store, so resolving it
# walks out of the tree that actually holds the declared test data.
_DATA = Path(__file__).parent / "fixtures" / "conflict_vlm"


def _messages(env) -> list[dict]:
    return env._effects.get("messages_sent", [])


# --------------------------------------------------------------------------- #
# R1: subagent_completion (delegate / baked, fired on `wait`)                   #
# --------------------------------------------------------------------------- #


def test_r1_subagent_completion_routes_reply_to_messages_sent():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env._background_events.append({
            "type": "subagent_completion",
            "trigger_time": 5,
            "agent_name": "Casey_Database_Admin",
            "mock_response": (
                "I have released the database write lock. You are clear to "
                "proceed."
            ),
            "state_effects": [],
            "reply_channel": "incident_war_room_channel",
        })
        env.step("wait 10")

        hits = [m for m in _messages(env) if m.get("sender") == "Casey_Database_Admin"]
        assert hits, f"subagent reply not in messages_sent: {_messages(env)}"
        assert "released the database write lock" in hits[0]["message"]
        assert hits[0]["channel"] == "incident_war_room_channel"

        # Also findable through the solver-facing `message search`.
        obs, _, _, _ = env.step(
            "message search --channel incident_war_room_channel --query lock"
        )
        assert "released the database write lock" in obs.command_output


def test_r1_message_send_state_effect_reaches_messages_sent():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env._background_events.append({
            "type": "subagent_completion",
            "trigger_time": 1,
            "agent_name": "Avery_Frontend_Engineer",
            "mock_response": "Done.",
            "state_effects": [
                "message send --channel 'operations_triage_channel' "
                "--target 'Devon' --message 'Avery here. Path owned by Devon "
                "Operations.'"
            ],
            "reply_channel": "operations_triage_channel",
        })
        env.step("wait 2")

        avery = " ".join(
            m.get("message", "")
            for m in _messages(env)
            if m.get("sender") == "Avery_Frontend_Engineer"
        )
        # Both the spoken summary and the explicit message-send effect land.
        assert "Path owned by Devon Operations" in avery
        assert "Done." in avery


# --------------------------------------------------------------------------- #
# R2: solver directly messages a subagent -> preset reply                       #
# --------------------------------------------------------------------------- #


def test_r2_solver_dm_elicits_subagent_reply():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        assert env._task is not None and env._task.data is not None
        env._task.data.private["mock_subagents"] = [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "Casey here. I have unlocked database writes.",
            "reply_channel": "incident_war_room_channel",
            "state_effects": [],
        }]
        env.step(
            "message send --channel incident_war_room_channel "
            "--target Casey_Database_Admin --message 'Casey, please unlock.'"
        )

        hits = [m for m in _messages(env) if m.get("sender") == "Casey_Database_Admin"]
        assert hits, f"Route 2 reply missing: {_messages(env)}"
        assert "unlocked database writes" in hits[0]["message"]


def test_routes_do_not_double_count():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env._task.data.private["mock_subagents"] = [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "Casey unlocking now.",
            "reply_channel": "incident_war_room_channel",
            "state_effects": [],
        }]
        # Route 2 (DM) then Route 1 (completion) with the SAME utterance/channel.
        env.step(
            "message send --channel incident_war_room_channel "
            "--target Casey_Database_Admin --message 'go'"
        )
        env._background_events.append({
            "type": "subagent_completion",
            "trigger_time": 1,
            "agent_name": "Casey_Database_Admin",
            "mock_response": "Casey unlocking now.",
            "state_effects": [],
            "reply_channel": "incident_war_room_channel",
        })
        env.step("wait 2")

        casey = [
            m
            for m in _messages(env)
            if m.get("sender") == "Casey_Database_Admin"
            and m.get("message") == "Casey unlocking now."
        ]
        assert len(casey) == 1, f"expected 1 deduped reply, got {len(casey)}"


# --------------------------------------------------------------------------- #
# L2 (M8/CL2, F6): an ``async_only`` sub-agent withholds its synchronous Route-2 #
# reply until ``wait`` fires its delayed completion (no-wait bypass fix)         #
# --------------------------------------------------------------------------- #


def test_async_only_subagent_withholds_synchronous_reply_until_delegate_then_wait():
    # The full M8/CL2 contract in one trajectory: an async_only decisive
    # sub-agent's reply arrives ONLY via the gold async playbook -- ``agent
    # delegate`` and THEN ``wait`` (see conflict_strict_validator
    # ``_gold_delegates_then_waits``). Neither a synchronous DM (L2b) nor a bare
    # ``wait`` after only messaging (L3) may elicit the decisive value.
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        assert env._task is not None and env._task.data is not None
        # The decisive sub-agent in a force_async_wait round: marked async_only
        # by the materializer and paired with a delayed completion event.
        env._task.data.private["mock_subagents"] = [{
            "agent_role": "Iris_SLA_Monitor",
            "mock_response": "Iris here. SLA verified: the P1 ticket was answered in time.",
            "reply_channel": "incident_war_room_channel",
            "virtual_duration_seconds": 30,
            "state_effects": [],
            "async_only": True,
        }]
        env._background_events.append({
            "type": "subagent_completion",
            "trigger_time": 30,
            "agent_name": "Iris_SLA_Monitor",
            "mock_response": "Iris here. SLA verified: the P1 ticket was answered in time.",
            "reply_channel": "incident_war_room_channel",
            "state_effects": [],
        })

        # L2b: the solver DMs the async_only agent (Route 2). Its reply must NOT
        # be mirrored synchronously (that would be stamped at virtual_time=0 and
        # bypass the sender-scoped within_window gate).
        env.step(
            "message send --channel incident_war_room_channel "
            "--target Iris_SLA_Monitor --message 'Iris, verify the SLA.'"
        )
        early = [m for m in _messages(env) if m.get("sender") == "Iris_SLA_Monitor"]
        assert not early, f"async_only reply leaked synchronously: {early}"

        # L3: a bare ``wait`` after only messaging (never delegating) still must
        # NOT deliver the decisive reply -- a DM that "merely narrates delegation"
        # is not the async playbook, so the within_window gate stays closed.
        env.step("wait 30")
        still_early = [m for m in _messages(env) if m.get("sender") == "Iris_SLA_Monitor"]
        assert not still_early, f"reply leaked without delegation: {still_early}"

        # The gold playbook: delegate, THEN wait past the completion. Only now
        # does the delayed reply land, stamped at the true trigger_time (the
        # within_window lower bound).
        env.step(
            "agent delegate --agent Iris_SLA_Monitor "
            "--instruction 'verify the SLA'"
        )
        env.step("wait 30")
        late = [m for m in _messages(env) if m.get("sender") == "Iris_SLA_Monitor"]
        assert late, f"delayed reply missing after delegate+wait: {_messages(env)}"
        assert "SLA verified" in late[0]["message"]
        assert late[0].get("virtual_time") == 30


def test_non_async_only_subagent_still_replies_synchronously():
    # Control: without ``async_only`` Route 2 still mirrors the reply immediately,
    # so only the opted-in decisive agent is gated (back-compat, E6).
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        assert env._task is not None and env._task.data is not None
        env._task.data.private["mock_subagents"] = [{
            "agent_role": "Harper_Support_Lead",
            "mock_response": "Harper here, picking it up now.",
            "reply_channel": "incident_war_room_channel",
            "state_effects": [],
        }]
        env.step(
            "message send --channel incident_war_room_channel "
            "--target Harper_Support_Lead --message 'status?'"
        )
        hits = [m for m in _messages(env) if m.get("sender") == "Harper_Support_Lead"]
        assert hits, f"non-async reply should mirror synchronously: {_messages(env)}"


# --------------------------------------------------------------------------- #
# L3 (M8/CL2, F6): an ``async_only`` decisive sub-agent's baked completion       #
# fires ONLY after the solver delegates -- a bare ``wait`` (no delegate) cannot  #
# elicit the decisive value, so the within_window gate fails closed.             #
# --------------------------------------------------------------------------- #


def _bake_async_only_iris(env) -> None:
    """Install an async_only decisive sub-agent + its baked completion (t=30)."""
    assert env._task is not None and env._task.data is not None
    env._task.data.private["mock_subagents"] = [{
        "agent_role": "Iris_SLA_Monitor",
        "mock_response": "Iris here. SLA verified within the window.",
        "reply_channel": "incident_war_room_channel",
        "virtual_duration_seconds": 30,
        "state_effects": [],
        "async_only": True,
    }]
    env._background_events.append({
        "type": "subagent_completion",
        "trigger_time": 30,
        "agent_name": "Iris_SLA_Monitor",
        "mock_response": "Iris here. SLA verified within the window.",
        "reply_channel": "incident_war_room_channel",
        "state_effects": [],
    })


def test_async_only_completion_requires_delegation():
    # No-delegate ablation: advancing the clock past the trigger with a bare
    # ``wait`` must NOT fire the async_only completion. The decisive value never
    # lands, so within_window has nothing in the sender-scoped window -> closed.
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        _bake_async_only_iris(env)
        env.step("wait 60")  # past trigger_time=30, but never delegated
        leaked = [m for m in _messages(env) if m.get("sender") == "Iris_SLA_Monitor"]
        assert not leaked, f"async_only completion fired without delegation: {leaked}"
        # Deferred, not dropped: the event stays pending for a later delegate+wait.
        assert any(
            e.get("agent_name") == "Iris_SLA_Monitor"
            for e in env._background_events
        ), "gated completion should remain pending until delegation"


def test_async_only_completion_fires_after_delegate_then_wait():
    # Genuine async playbook: once the solver delegates, the next ``wait`` fires
    # the completion, stamped at the true trigger_time (the within_window lower
    # bound) so the decisive reply grades correctly.
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        _bake_async_only_iris(env)
        env.step(
            "agent delegate --agent Iris_SLA_Monitor "
            "--instruction 'verify the SLA'"
        )
        env.step("wait 60")
        arrived = [m for m in _messages(env) if m.get("sender") == "Iris_SLA_Monitor"]
        assert arrived, f"delegated async_only reply missing after wait: {_messages(env)}"
        assert "SLA verified" in arrived[0]["message"]
        assert arrived[0].get("virtual_time") == 30


def test_async_only_delegation_matches_short_agent_name():
    # The gate resolves delegation by the same normalized substring rule as
    # AgentManager: delegating with a short handle ("Iris") clears the gate for
    # the full-role completion event ("Iris_SLA_Monitor").
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        _bake_async_only_iris(env)
        env.step("agent delegate --agent Iris --instruction 'verify'")
        env.step("wait 60")
        arrived = [m for m in _messages(env) if m.get("sender") == "Iris_SLA_Monitor"]
        assert arrived, f"short-name delegation should clear the gate: {_messages(env)}"


def test_non_async_only_completion_still_fires_on_bare_wait():
    # Back-compat (E6): a NON-async_only baked completion still fires on a bare
    # ``wait`` with no delegation -- only opted-in decisive agents are L3-gated.
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        assert env._task is not None and env._task.data is not None
        env._task.data.private["mock_subagents"] = [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "Casey unlocking now.",
            "reply_channel": "incident_war_room_channel",
            "virtual_duration_seconds": 5,
            "state_effects": [],
        }]
        env._background_events.append({
            "type": "subagent_completion",
            "trigger_time": 5,
            "agent_name": "Casey_Database_Admin",
            "mock_response": "Casey unlocking now.",
            "reply_channel": "incident_war_room_channel",
            "state_effects": [],
        })
        env.step("wait 10")
        hits = [m for m in _messages(env) if m.get("sender") == "Casey_Database_Admin"]
        assert hits, f"non-async_only completion should fire on bare wait: {_messages(env)}"


# --------------------------------------------------------------------------- #
# Grading: EffectChecker surfaces sender for attribution                        #
# --------------------------------------------------------------------------- #


def test_effect_checker_surfaces_sender_in_transcript(monkeypatch):
    import clasharness.evaluation.llm_judge as judge_mod

    captured = {}

    class _R:
        passed = True
        reason = "ok"
        ok = True
        error = ""

    def _fake_judge(transcript, criterion, model=None):
        captured["transcript"] = transcript
        return _R()

    monkeypatch.setattr(judge_mod, "judge_majority", _fake_judge)

    from clasharness.evaluation.checkers.effect_checker import EffectChecker

    chk = EffectChecker(
        "messages_sent",
        "field_semantic_match",
        {
            "field": "message",
            "criterion": "Casey_Database_Admin acknowledges and agrees to unlock.",
        },
    )
    env_state = {
        "effects": {
            "messages_sent": [
                {
                    "sender": "Casey_Database_Admin",
                    "target": "incident_war_room_channel",
                    "message": "I have released the write lock.",
                }
            ]
        }
    }
    res = chk.evaluate(env_state, {})
    assert "From Casey_Database_Admin" in captured["transcript"]
    assert res.passed


# --------------------------------------------------------------------------- #
# G1: real-round recompile reconciles channels + stamps reply_channel           #
# --------------------------------------------------------------------------- #


def _load_conflict_spec(slug: str):
    from clasharness.scenario.conflict_vlm_spec import (
        ConflictVLMScenarioSpec,
        ConflictVLMSeed,
    )

    spec_path = _DATA / slug / f"{slug}_spec.json"
    spec = ConflictVLMScenarioSpec.read_json(spec_path)
    raw = json.loads(spec_path.read_text())
    seed = ConflictVLMSeed(
        slug=raw.get("scenario_slug", slug),
        domain=raw.get("domain", ""),
        title=raw.get("title", ""),
        target_rounds=len(spec.rounds),
        core_conflict=raw.get("core_conflict", ""),
        conflict_taxonomy=list(raw.get("conflict_taxonomy", []) or []),
        participants=[],
        required_constraints=[],
        allowed_tool_families=list(raw.get("allowed_tool_families", []) or []),
    )
    return seed, spec


def test_g1_recompile_reconciles_channels_and_sets_reply_channel():
    from clasharness.scenario.conflict_vlm_materializer import (
        _compile_round_to_task,
        _session_channel_registry,
    )

    slug = "incident_commander_selection_v01"
    seed, spec = _load_conflict_spec(slug)
    # The graded channel identity is the session DISPLAY name ('#incident-war-
    # room'), not the 'session_id' storage key: criteria and the solver's
    # `message search` both speak display names.
    session_channels = set(_session_channel_registry(spec.sessions))
    assert session_channels, "scenario must declare sessions"

    round_spec = spec.rounds[0]
    round_updates = [u for u in spec.updates if u.update_id in round_spec.update_ids]
    task = _compile_round_to_task(
        seed, round_spec, spec.evals, round_updates, round_spec.mock_subagents, spec
    )

    mocks = task.data.private["mock_subagents"]
    assert mocks, "round must have mock subagents"
    for m in mocks:
        # Every subagent gets a canonical reply channel...
        assert m.get("reply_channel") in session_channels, m
        # ...and no LLM-invented channel survives in its effects: whatever the
        # authoring call made up is rewritten onto a shipped display name.
        for eff in m.get("state_effects", []):
            if isinstance(eff, str) and "--channel" in eff:
                value = eff.split("--channel", 1)[1].split("--", 1)[0].strip()
                value = value.strip("'\"")
                assert value in session_channels, eff

    completions = [
        e
        for e in task.data.private["background_events"]
        if e.get("type") == "subagent_completion"
    ]
    assert completions, "expected baked subagent_completion events"
    for e in completions:
        assert e.get("reply_channel") in session_channels, e


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
