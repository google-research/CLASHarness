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

"""Tests for the M1/M2/M3 seed difficulty knobs.

Each knob is a no-op unless its seed opts in, and each has one property that
has to hold or the knob is worthless:

  M1 -- the assignment is deterministic and hits the target mix, and the
        UNFORCED prompt text is byte-identical to what shipped before (E6).
  M2 -- the injected decoy can never satisfy the round's own true-value check.
        That is the whole safety argument; if it fails, the decoy silently
        marks wrong answers correct.
  M3 -- stripping covers BOTH channels (the round instruction and the sub-agent
        replies), never removes evidence a graded check depends on, and never
        empties either channel. The instruction is the channel that matters:
        88.0% of asset-locus rounds name the asset there vs 71.4% in a reply,
        and only 2.8% name it in a reply ALONE -- so a reply-only strip had a
        reachable ceiling of 17 rounds in 598. This is a correctness fix, not a
        difficulty lever: see ``_strip_signposts`` for why the -9.4pp claim that
        originally motivated it was withdrawn.
"""

import collections
from pathlib import Path
import unittest

import clasharness
from clasharness.scenario import conflict_vlm_materializer
from clasharness.scenario.conflict_llm_pipeline import _assigned_round_type
from clasharness.scenario.conflict_llm_pipeline import _format_round_type_directive
from clasharness.scenario.conflict_llm_pipeline import _round_type_pattern
from clasharness.scenario.conflict_vlm_materializer import _ASSET_LIST_GUIDANCE
from clasharness.scenario.conflict_vlm_materializer import _asset_mention_terms
from clasharness.scenario.conflict_vlm_materializer import _decoy_value_for
from clasharness.scenario.conflict_vlm_materializer import _inject_decoy
from clasharness.scenario.conflict_vlm_materializer import _strip_signposts
from clasharness.scenario.conflict_vlm_materializer import (
    _true_value_is_recoverable_from,
)
from clasharness.scenario.conflict_vlm_spec import ConflictVLMRoundSpec
from clasharness.scenario.conflict_vlm_spec import ConflictVLMScenarioSpec
from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.conflict_vlm_spec import load_conflict_vlm_seed_file
from clasharness.scenario.spec import MockSubagent
from clasharness.scenario.spec import ScenarioWorkspace
from clasharness.scenario.vlm_spec import VLMGenerationPolicy
from clasharness.scenario.vlm_spec import VLMInputPolicy

# Anchored on the package, not the working directory, so the test resolves the
# same way under pytest, unittest, and the build system.
_SEED_DIR = (
    # No `.resolve()`: under the build system `__file__` is a symlink into the
    # CAS, and resolving it walks out of the runfiles tree.
    Path(clasharness.__file__).parent
    / "data"
    / "scenario_seeds"
    / "conflict_vlm_v1"
)


def _round(round_id="r01", resolved_value="LAT-8821", locus="asset:ops_console_panel"):
  return ConflictVLMRoundSpec.from_dict({
      "round_id": round_id,
      "round_type": "action",
      "resolved_value": resolved_value,
      "resolved_value_locus": locus,
  })


class TestSeedKnobDefaults(unittest.TestCase):

  def test_all_three_default_off(self):
    seed = ConflictVLMSeed.from_dict({"slug": "x", "target_rounds": 3})
    self.assertFalse(seed.force_round_type_plan)
    self.assertFalse(seed.force_decoy)
    self.assertFalse(seed.force_no_signpost)

  def test_from_dict_parses_all_three(self):
    seed = ConflictVLMSeed.from_dict({
        "slug": "x",
        "target_rounds": 3,
        "force_round_type_plan": True,
        "force_decoy": True,
        "force_no_signpost": True,
    })
    self.assertTrue(seed.force_round_type_plan)
    self.assertTrue(seed.force_decoy)
    self.assertTrue(seed.force_no_signpost)


class TestM1RoundTypePlan(unittest.TestCase):

  def test_pattern_hits_target_mix_exactly(self):
    counts = collections.Counter(_round_type_pattern())
    self.assertEqual(counts["investigation"], 18)
    self.assertEqual(counts["decision"], 10)
    self.assertEqual(counts["action"], 10)
    self.assertEqual(counts["verification"], 2)

  def test_pattern_is_interleaved_not_blocked(self):
    # A scenario consumes a WINDOW of the pattern, so long runs of one type
    # would hand some scenarios ten straight 'investigation' rounds.
    pattern = _round_type_pattern()
    longest = 1
    current = 1
    for i in range(1, len(pattern)):
      current = current + 1 if pattern[i] == pattern[i - 1] else 1
      longest = max(longest, current)
    self.assertLessEqual(longest, 2)

  def test_assignment_is_stable_and_not_process_salted(self):
    # Must survive a checkpoint resume, so it cannot use the builtin ``hash``.
    first = [_assigned_round_type("some_seed", i) for i in range(1, 11)]
    second = [_assigned_round_type("some_seed", i) for i in range(1, 11)]
    self.assertEqual(first, second)
    self.assertEqual(
        _assigned_round_type("some_seed", 1),
        _round_type_pattern()[
            (int(__import__("hashlib").sha1(b"some_seed").hexdigest(), 16) + 1) % 40
        ],
    )

  def test_different_seeds_get_different_sequences(self):
    a = [_assigned_round_type("seed_alpha", i) for i in range(1, 11)]
    b = [_assigned_round_type("seed_beta", i) for i in range(1, 11)]
    self.assertNotEqual(a, b)

  def test_unforced_directive_is_byte_identical_to_shipped_text(self):
    # E6: a seed that has not opted in must produce the historical prompt.
    text = _format_round_type_directive(None)
    self.assertTrue(text.startswith("Choose a 'round_type' that fits this beat"))
    self.assertIn("62.8% 'action'", text)
    self.assertNotIn("ASSIGNED", text)

  def test_forced_directive_names_the_type_and_demands_the_shape(self):
    text = _format_round_type_directive("investigation")
    self.assertIn("ASSIGNED: 'investigation'", text)
    # The label is inert downstream, so the directive has to drive the SHAPE.
    self.assertIn("WORK OUT what is true", text)
    self.assertIn("do not write some other kind of round", text)


class TestM2Decoy(unittest.TestCase):

  def test_decoy_never_satisfies_the_true_value_check(self):
    # The one invariant that matters: an agent reporting the decoy must FAIL.
    for value in [
        "SIG-APP-STAGE-8821",
        "1,820 ms",
        "94.2%",
        "SEC-9942-Z",
        "PROV-JOB-8842-OK",
        "SCOPE HOLD (2 of 3 approvers unaligned)",
        "5",
        "0",
        "999",
    ]:
      with self.subTest(value=value):
        decoy = _decoy_value_for(value)
        if decoy is None:
          continue
        self.assertNotEqual(decoy, value)
        self.assertFalse(_true_value_is_recoverable_from(value, decoy))

  def test_non_numeric_values_are_skipped_not_guessed(self):
    self.assertIsNone(_decoy_value_for("HOLD_FOR_RECONCILIATION"))
    self.assertIsNone(_decoy_value_for(""))

  def test_injection_updates_both_the_reply_and_the_background_event(self):
    # The background event is the copy the environment replays; updating only
    # ``mock_subagents`` would inject a decoy the solver never sees.
    subs = [{"agent_role": "Dana_DBA", "mock_response": "Pool looks hot."}]
    events = [{
        "type": "subagent_completion",
        "agent_name": "Dana_DBA",
        "mock_response": "Pool looks hot.",
    }]
    decoy = _inject_decoy(subs, events, _round(resolved_value="LAT-8821"), "slug")
    self.assertIsNotNone(decoy)
    self.assertIn(decoy, subs[0]["mock_response"])
    self.assertEqual(events[0]["mock_response"], subs[0]["mock_response"])

  def test_injection_is_deterministic(self):
    def run():
      subs = [
          {"agent_role": "A", "mock_response": "one"},
          {"agent_role": "B", "mock_response": "two"},
      ]
      _inject_decoy(subs, [], _round(), "slug")
      return [s["mock_response"] for s in subs]

    self.assertEqual(run(), run())

  def test_no_subagents_is_a_noop(self):
    self.assertIsNone(_inject_decoy([], [], _round(), "slug"))


class TestM3SignpostStripping(unittest.TestCase):

  def test_mention_terms_cover_id_and_prose_forms(self):
    terms = _asset_mention_terms("lock_manager_console_concurrency_status_panel")
    self.assertIn("lock_manager_console_concurrency_status_panel", terms)
    self.assertIn("lock manager console", terms)

  def test_strips_the_sentence_that_names_the_decisive_asset(self):
    subs = [{
        "agent_role": "A",
        "mock_response": (
            "The lease did not release cleanly. Please inspect the lock manager"
            " console to read the current holder. I am standing by."
        ),
    }]
    res = _strip_signposts(subs, [], "lock_manager_console", [])
    self.assertEqual(len(res.removed), 1)
    self.assertEqual(res.outcome, "stripped")
    self.assertEqual(res.channel, "mock")
    self.assertNotIn("lock manager console", subs[0]["mock_response"].lower())
    self.assertIn("did not release cleanly", subs[0]["mock_response"])

  def test_rolls_back_when_stripping_would_break_a_graded_check(self):
    # The signposting sentence also carries the value a check greps for, so
    # removing it would make the gold trajectory ungradeable.
    subs = [{
        "agent_role": "A",
        "mock_response": (
            "Status nominal. The lock manager console reads LAT-8821 for the"
            " current holder."
        ),
    }]
    checks = [{
        "type": "effect",
        "effect_type": "messages_sent",
        "condition": "field_contains",
        "expected": {"field": "content", "value": "LAT-8821"},
    }]
    res = _strip_signposts(subs, [], "lock_manager_console", checks)
    self.assertEqual(res.removed, [])
    # Distinguished from a plain no-op: the knob WANTED to fire here.
    self.assertEqual(res.outcome, "rolled_back_check")
    self.assertEqual(res.channel, "")
    self.assertIn("LAT-8821", subs[0]["mock_response"])

  def test_never_leaves_a_reply_empty(self):
    # An emptied reply is a broken artifact AND would void the M8/CL2 async
    # check, which binds its temporal window to a sub-agent's reply.
    subs = [{
        "agent_role": "A",
        "mock_response": "Please check the lock manager console.",
    }]
    res = _strip_signposts(subs, [], "lock_manager_console", [])
    self.assertEqual(res.removed, [])
    self.assertEqual(res.outcome, "rolled_back_empty")
    self.assertTrue(subs[0]["mock_response"].strip())

  def test_strip_is_all_or_nothing_across_subagents(self):
    # One un-strippable reply rolls back the WHOLE reply channel, so the
    # scenario never ships half its sub-agents de-signposted.
    subs = [
        {"agent_role": "A", "mock_response": "Look at the lock manager console."},
        {
            "agent_role": "B",
            "mock_response": (
                "Queue is draining. The lock manager console shows the holder."
            ),
        },
    ]
    before = [s["mock_response"] for s in subs]
    res = _strip_signposts(subs, [], "lock_manager_console", [])
    self.assertEqual(res.removed, [])
    self.assertEqual(res.outcome, "rolled_back_empty")
    self.assertEqual([s["mock_response"] for s in subs], before)

  def test_noop_when_nothing_mentions_the_asset(self):
    subs = [{"agent_role": "A", "mock_response": "All clear on my side."}]
    res = _strip_signposts(
        subs, [], "lock_manager_console", [], "Resolve the contention."
    )
    self.assertEqual(res.removed, [])
    # NOT a rollback -- there was simply nothing to strip. The metadata stamp
    # has to keep these apart or the M3 effect size is computed over the wrong
    # denominator.
    self.assertEqual(res.outcome, "no_mention")
    self.assertEqual(res.channel, "")
    self.assertEqual(subs[0]["mock_response"], "All clear on my side.")
    # The instruction is echoed back unchanged, so the caller can assign it
    # unconditionally.
    self.assertEqual(res.instruction, "Resolve the contention.")

  # --- the instruction channel (the dominant one) ---------------------------

  _NAMED = "Read the lock manager console to get the current holder."
  _PLAIN = "Two writers are contending and the queue is backed up."

  def test_strips_the_instruction_even_when_no_reply_signposts(self):
    # The case that made the reply-only implementation near-inert: 88.1% of
    # asset-locus rounds name the asset in the instruction, and only 0.8% name
    # it in a reply WITHOUT also naming it in the instruction.
    subs = [{"agent_role": "A", "mock_response": self._PLAIN}]
    res = _strip_signposts(
        subs, [], "lock_manager_console", [], "Resolve it. " + self._NAMED
    )
    self.assertEqual(res.outcome, "stripped")
    self.assertEqual(res.channel, "instruction")
    self.assertEqual(res.instruction_removed, 1)
    self.assertNotIn("lock manager console", res.instruction.lower())
    # The round still states a task; de-signposting must not blank it out.
    self.assertIn("Resolve it.", res.instruction)
    self.assertEqual(subs[0]["mock_response"], self._PLAIN)

  def test_strips_both_channels_when_both_signpost(self):
    subs = [
        {"agent_role": "A", "mock_response": self._PLAIN + " " + self._NAMED},
        {"agent_role": "B", "mock_response": self._PLAIN},
    ]
    res = _strip_signposts(
        subs, [], "lock_manager_console", [], "Resolve it. " + self._NAMED
    )
    self.assertEqual(res.outcome, "stripped")
    self.assertEqual(res.channel, "both")
    self.assertEqual(res.instruction_removed, 1)
    self.assertEqual(subs[0]["mock_response"], self._PLAIN)
    self.assertNotIn("lock manager console", res.instruction.lower())

  def test_an_unstrippable_instruction_does_not_forfeit_the_reply_strip(self):
    # Emptiness rollback is PER CHANNEL. An instruction that is nothing but its
    # signpost cannot be stripped, but that is no reason to also keep the
    # signpost in the replies.
    subs = [{"agent_role": "A", "mock_response": self._PLAIN + " " + self._NAMED}]
    res = _strip_signposts(subs, [], "lock_manager_console", [], self._NAMED)
    self.assertEqual(res.outcome, "stripped")
    self.assertEqual(res.channel, "mock")
    self.assertEqual(res.instruction_removed, 0)
    self.assertEqual(res.instruction, self._NAMED)
    self.assertEqual(subs[0]["mock_response"], self._PLAIN)

  def test_asset_list_guidance_is_never_stripped(self):
    # The guidance names no asset, and it is the fairness guarantee: it tells
    # the agent the discovery primitive exists, which is what keeps a
    # de-signposted round a search rather than a guess.
    instruction = "Resolve it. " + self._NAMED + "\n\n" + _ASSET_LIST_GUIDANCE
    subs = [{"agent_role": "A", "mock_response": self._PLAIN}]
    res = _strip_signposts(subs, [], "lock_manager_console", [], instruction)
    self.assertEqual(res.outcome, "stripped")
    self.assertIn(_ASSET_LIST_GUIDANCE, res.instruction)
    self.assertNotIn("lock manager console", res.instruction.lower())
    # And it keeps its own paragraph, so a de-signposted instruction is not
    # also a differently-formatted one.
    self.assertEqual(res.instruction, "Resolve it.\n\n" + _ASSET_LIST_GUIDANCE)

  def test_instruction_left_as_bare_guidance_rolls_that_channel_back(self):
    # Guidance alone states no task. That is not a hard round, it is a broken
    # one.
    instruction = self._NAMED + "\n\n" + _ASSET_LIST_GUIDANCE
    subs = [{"agent_role": "A", "mock_response": self._PLAIN}]
    res = _strip_signposts(subs, [], "lock_manager_console", [], instruction)
    self.assertEqual(res.outcome, "rolled_back_empty")
    self.assertEqual(res.instruction, instruction)

  def test_a_check_literal_living_only_in_the_instruction_rolls_back_the_round(
      self,
  ):
    # The gold-solvability gate has to see the instruction too, or a check whose
    # value appears ONLY in the task text becomes silently ungradeable. The
    # rollback is whole-round: the two strips are validated together, so only
    # their combination is known to be safe.
    instruction = "Resolve it. The lock manager console reads LAT-8821."
    checks = [{
        "type": "effect",
        "effect_type": "messages_sent",
        "condition": "field_contains",
        "expected": {"field": "content", "value": "LAT-8821"},
    }]
    subs = [{"agent_role": "A", "mock_response": self._PLAIN + " " + self._NAMED}]
    res = _strip_signposts(subs, [], "lock_manager_console", checks, instruction)
    self.assertEqual(res.outcome, "rolled_back_check")
    self.assertEqual(res.channel, "")
    self.assertEqual(res.instruction, instruction)
    self.assertEqual(subs[0]["mock_response"], self._PLAIN + " " + self._NAMED)


class TestKnobOutcomeIsRecordedInTheArtifact(unittest.TestCase):
  """The round must carry a record of what the knobs actually did.

  Without it the measurement round cannot produce an effect size: a round that
  was successfully de-signposted is textually identical to one that never had a
  signpost, so "the knob fired here" is unrecoverable after the fact.
  """

  def _compile(self, instruction="", **knobs):
    seed = ConflictVLMSeed(
        slug="knob_probe",
        domain="system",
        title="Knob Probe",
        target_rounds=1,
        core_conflict="Mock conflict",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        **knobs,
    )
    round_spec = ConflictVLMRoundSpec(
        round_id="r01",
        round_type="action",
        visible_asset_ids=[],
        required_modalities=[],
        expected_anchor_ids=[],
        required_tool_families=["message"],
        resolved_value="LAT-8821",
        resolved_value_locus="asset:lock_manager_console",
        instruction=instruction,
    )
    spec = ConflictVLMScenarioSpec(
        series="conflict_vlm_v1",
        scenario_slug="knob_probe",
        title="Knob Probe",
        domain="system",
        core_conflict="Mock conflict",
        conflict_taxonomy=["temporal_staleness"],
        modalities=[],
        input_policy=VLMInputPolicy(),
        generation_policy=VLMGenerationPolicy(),
        allowed_tool_families=["agents", "file", "message"],
        assets=[],
        updates=[],
        rounds=[round_spec],
        evals=[],
        workspace=ScenarioWorkspace(),
        sessions=[],
    )
    mock = MockSubagent(
        agent_role="Dana_DBA",
        mock_response=(
            "The lease did not release cleanly. Please inspect the lock manager"
            " console to read the current holder."
        ),
    )
    return conflict_vlm_materializer._compile_round_to_task(
        seed=seed,
        round_spec=round_spec,
        evals=[],
        round_updates=[],
        round_mocks=[mock],
        spec=spec,
    )

  def test_no_knobs_leaves_ground_truth_metadata_untouched(self):
    # Opted-out seeds must ship exactly what they shipped before.
    task = self._compile()
    self.assertNotIn("difficulty_knobs", task.ground_truth.metadata)

  def test_m3_records_that_it_actually_fired(self):
    task = self._compile(force_no_signpost=True)
    rec = task.ground_truth.metadata["difficulty_knobs"]["no_signpost"]
    self.assertTrue(rec["applied"])
    self.assertEqual(rec["outcome"], "stripped")
    self.assertEqual(rec["asset"], "lock_manager_console")
    self.assertEqual(rec["removed_count"], 1)
    # Which channel fired is not recoverable from the shipped corpus after the
    # fact, and "the instruction was stripped" vs "the instruction never named
    # it" mean opposite things when the effect is attributed.
    self.assertEqual(rec["channel"], "mock")
    self.assertEqual(rec["instruction_removed_count"], 0)

  def test_m3_designposts_the_shipped_task_instruction(self):
    # The regression guard for the defect that made this knob near-inert: the
    # strip used to rewrite only the mock replies, so the round shipped with its
    # own instruction still naming the asset. ``task.instruction`` is what the
    # agent reads, so that is what this asserts.
    task = self._compile(
        instruction="Resolve the contention. Read the lock manager console.",
        force_no_signpost=True,
    )
    self.assertNotIn("lock manager console", task.instruction.lower())
    self.assertIn("Resolve the contention.", task.instruction)
    rec = task.ground_truth.metadata["difficulty_knobs"]["no_signpost"]
    self.assertEqual(rec["channel"], "both")
    self.assertEqual(rec["instruction_removed_count"], 1)

  def test_m3_leaves_the_instruction_alone_when_the_seed_opts_out(self):
    instruction = "Resolve the contention. Read the lock manager console."
    task = self._compile(instruction=instruction)
    self.assertEqual(task.instruction, instruction)

  def test_m2_records_the_decoy_and_the_true_value(self):
    task = self._compile(force_decoy=True)
    rec = task.ground_truth.metadata["difficulty_knobs"]["decoy"]
    self.assertTrue(rec["applied"])
    self.assertEqual(rec["true_value"], "LAT-8821")
    self.assertNotEqual(rec["value"], "LAT-8821")

  def test_the_record_never_reaches_the_agent(self):
    # ``ground_truth`` is grader-side; a knob record visible to the solver would
    # hand it the decoy answer and the name of the asset we just hid.
    task = self._compile(force_decoy=True, force_no_signpost=True)
    self.assertNotIn("difficulty_knobs", task.data.public)
    self.assertNotIn("difficulty_knobs", task.data.private)


class TestSeedCorpusAllocation(unittest.TestCase):
  """The shipped seeds must carry exactly the reviewed allocation."""

  def _seeds(self):
    files = sorted(_SEED_DIR.glob("*.json"))
    self.assertTrue(files, f"no seeds found under {_SEED_DIR}")
    return {s.slug: s for s in (load_conflict_vlm_seed_file(p) for p in files)}

  def test_allocation_counts(self):
    # These counts pin the state the SHIPPED EVAL CORPUS was generated from
    # (data/scenarios/conflict_vlm_v3_3.6_flash, generated 07-30 18:50-19:49),
    # independently corroborated from the corpus itself: exactly one seed
    # (sla_support_response) emits the M8/CL2 delegate->wait gold trajectory and
    # the injected ``within_window`` check (30/30 tasks), and every other seed
    # emits neither (0/30) -- which the generation-time hard gate
    # (``_has_delegate_then_wait``) makes impossible if its flag were set.
    #
    # DO NOT raise these to match a proposed allocation. Widening a knob here
    # without regenerating the corpus produces a seed corpus that silently
    # disagrees with the measurements attributed to it; that exact drift
    # (14/1/0/0/0 -> 20/8/11/8/7, uncaught) is why this assertion is now exact
    # rather than a lower bound.
    seeds = self._seeds()
    self.assertEqual(len(seeds), 20)
    self.assertEqual(sum(s.force_vision_locus for s in seeds.values()), 14)
    self.assertEqual(sum(s.force_async_wait for s in seeds.values()), 1)
    self.assertEqual(sum(s.force_round_type_plan for s in seeds.values()), 0)
    self.assertEqual(sum(s.force_decoy for s in seeds.values()), 0)
    self.assertEqual(sum(s.force_no_signpost for s in seeds.values()), 0)

  def test_benchmark_and_reverse_discriminating_seeds_are_untouched(self):
    # sla_support_response is the standout (13.3% pass / +21.0pp gap) and the
    # two reverse-discriminating seeds are the deliberately UNTREATED control
    # arm; changing any of them destroys the reference points every difficulty
    # measurement is compared against.
    #
    # Asserted as an exact per-flag map, not a set of assertFalse calls: the
    # previous version checked only M1/M2/M3, so a ``force_vision_locus``
    # insertion into both control-arm seeds passed this guard unnoticed. Every
    # knob that exists must appear here, including the ones these seeds
    # legitimately carry (sla is the sole M8/CL2 opt-in).
    expected = {
        "sla_support_response": {
            "force_vision_locus": True,
            "force_async_wait": True,
            "force_round_type_plan": False,
            "force_decoy": False,
            "force_no_signpost": False,
        },
        "clinical_trial_dosage_conflict": {
            "force_vision_locus": False,
            "force_async_wait": False,
            "force_round_type_plan": False,
            "force_decoy": False,
            "force_no_signpost": False,
        },
        "payment_chargeback_dispute": {
            "force_vision_locus": False,
            "force_async_wait": False,
            "force_round_type_plan": False,
            "force_decoy": False,
            "force_no_signpost": False,
        },
    }
    seeds = self._seeds()
    for slug, flags in expected.items():
      s = seeds[slug]
      for flag, want in flags.items():
        with self.subTest(slug=slug, flag=flag):
          self.assertEqual(getattr(s, flag), want)


if __name__ == "__main__":
  unittest.main()
