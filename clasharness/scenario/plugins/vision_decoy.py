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

"""Vision-decoy conflict plugin: a confident subagent asserts a value that
contradicts the decisive visual evidence.

This is a *selective* difficulty mechanic, routed only to seeds that opt in by
listing the ``vision_decoy`` token in their ``difficulty_challenges`` (or
``conflict_taxonomy``). It composes with the ``force_vision_locus`` seed knob:
that knob confines the round's decisive value to a visual asset (locus
``asset:<id>``) and makes opening that asset a graded (``required``) step, while
this plugin adds the *social pressure* not to bother — a teammate who
confidently reports a plausible but WRONG value (a "decoy"). The solver passes
only by trusting the pixels over the confident assertion.

Because it attaches per-token, it does not homogenize the corpus: seeds that
don't carry the token keep their native conflict archetype untouched.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class VisionDecoyPlugin(BaseConflictPlugin):
  """Plugin modeling a confident-but-wrong assertion that contradicts the image."""

  SPEC = ConflictPluginSpec(
      name="vision_decoy",
      summary=(
          "A teammate confidently asserts a plausible but WRONG value (a decoy) "
          "for the round's decisive fact, contradicting the visual asset that "
          "actually holds the true value. The solver must trust the pixels over "
          "the confident assertion and act on the image's value, not the decoy."
      ),
      category="challenge",
      hooks=("layer0", "layer1", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "vision_decoy"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "VISION DECOY: In the decisive round(s), one teammate must CONFIDENTLY"
        " assert a specific, plausible-but-WRONG value (a 'decoy') for the fact"
        " the round turns on (e.g. an amount, code, status, count, or"
        " timestamp), stating it as already-confirmed ('I already checked — it's"
        " <decoy>'). The TRUE value must live ONLY in the decisive visual asset;"
        " the decoy value must be clearly different from the true value and must"
        " NOT be presented anywhere as the correct answer. The tension is"
        " social: it is tempting to relay the confident teammate's decoy instead"
        " of opening the image."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "VISION DECOY preference_rule: the authoritative source is the decisive"
        " visual asset, NOT any teammate's spoken/written claim. Model a subagent"
        " who reports the decoy value with high confidence and refuses to revise"
        " it until the coordinator reads the image and cites the true value back."
        " The coordinator must override the decoy using only what the pixels show."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "VISION DECOY CHALLENGE: the expert trajectory MUST open the decisive"
        " visual asset, detect that the teammate's confidently-asserted decoy"
        " value contradicts the image, and then act on / report the value read"
        " from the image — never the decoy. Do not shortcut by relaying the"
        " teammate's claim."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "VISION DECOY EVAL: grade that the coordinator's final action and report"
        " use the TRUE value read from the decisive visual asset, and that the"
        " decoy value asserted by the teammate does NOT appear as the accepted"
        " answer. Opening the decisive asset is a required step for the round."
    )
