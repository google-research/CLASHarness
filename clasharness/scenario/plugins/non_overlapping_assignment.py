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

"""Non-overlapping review assignment + document sign-gate plugin (M4, #5).

Steers the generator to build a collaborative document review in which a single
document (a contract, a launch plan, a design) can be signed off only once
**every distinct review aspect has been cleared by its own reviewer**. The work
is partitioned into non-overlapping aspects — legal, privacy, security,
business, layout — each owned by exactly one reviewer, so no two reviewers grade
the same aspect and no aspect is left unowned. The document's sign gate stays
shut until all aspects are ``ok`` with no outstanding blockers. The
tempting-but-wrong moves are signing prematurely while an aspect is still
pending or blocked, or one reviewer overreaching into another's aspect.

The mechanic rides the ``review`` family (with ``lock`` available to model
exclusive, non-overlapping ownership of each aspect/section): ``review assign``
routes an aspect to a reviewer, ``review submit --aspect A --status ok|blocked``
records that reviewer's verdict and emits ``review_submitted``, and ``review
sign --document D`` is the gate — it emits ``document_signed`` only when at
least one aspect has been reviewed and every aspect in play is ``ok`` with no
blockers, failing otherwise. Because a blocked or still-pending aspect keeps the
gate shut, full credit requires collecting and clearing every reviewer's verdict
before signing.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class NonOverlappingAssignmentPlugin(BaseConflictPlugin):
  """Plugin modeling a document signed only when every distinct review aspect clears."""

  SPEC = ConflictPluginSpec(
      name="non_overlapping_assignment",
      summary=(
          "A document is signed off only once every distinct review aspect "
          "(legal, privacy, security, business, layout) has been cleared by its "
          "own reviewer; work is partitioned into non-overlapping aspects, and "
          "signing before every aspect is ok — or a reviewer grading an aspect "
          "that is not theirs — is the failure mode."
      ),
      category="subagent_collaboration",
      skill_families=("review", "lock"),
      emits_effects=("document_signed",),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "non_overlapping_assignment"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery that a single document cannot be "
        "signed off by one pass but needs MULTIPLE distinct review aspects "
        "(e.g. legal, privacy, security, business, layout), each owned by a "
        "different reviewer with NO overlap; and 2) Coordinating the sign-off "
        "so every aspect is cleared before the document is signed, rather than "
        "signing prematurely while an aspect is still pending or blocked."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model subagents/stakeholders who each own exactly ONE review aspect "
        "(one is legal, one is privacy, one is security, one is business) and "
        "who report back at different times, so a coordinator that signs after "
        "the first one or two verdicts skips the rest. At least one reviewer "
        "should return a BLOCKED verdict with a concrete issue that must be "
        "resolved (re-submitted as ok) before signing. A stakeholder should "
        "press to sign early for the loudest 'looks good', and another should "
        "be tempted to grade an aspect that is not theirs — both are wrong."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create workspace files that make the non-overlapping assignment "
        "legible: an assignment matrix mapping each aspect to exactly one "
        "reviewer (and showing the aspects do not overlap), per-aspect review "
        "notes capturing each verdict and any blocking issue, and a sign-off "
        "policy stating the document may be signed ONLY once every aspect is "
        "ok with no outstanding blockers."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- NON-OVERLAPPING REVIEW CHALLENGE: The trajectory assigns each aspect "
        "to a distinct reviewer (`review assign --aspect <a> --reviewer <r>`), "
        "collects a verdict per aspect (`review submit --aspect <a> --status "
        "ok|blocked [--issue <text>]`, each emitting one `review_submitted`), "
        "resolves any blocked aspect by re-submitting it as ok, and only THEN "
        "signs (`review sign --document <d>`), which emits `document_signed`. A "
        "`review sign` attempted while any aspect is still pending or blocked "
        "MUST fail and emit nothing, so the document is signed only after every "
        "distinct aspect has been cleared."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- EVALS: Add an effect check asserting the document was signed only "
        "after a complete review, i.e. an `exists` assertion on a "
        "`document_signed` effect together with a `count_gte`-style assertion "
        "that multiple distinct `review_submitted` effects were emitted (one "
        "per reviewed aspect). A run that signed with aspects still "
        "pending/blocked, or that reviewed only a single aspect, must NOT pass."
    )
