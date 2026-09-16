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

"""Time-bounded window and invitation session expiration conflict plugin.

Mechanized on the ``booking`` and ``pairing`` families. The conflict is "a
tentative hold or session dies unless the coordinated action lands before its
window closes", and both skills enforce exactly that rather than merely
narrating it: each record carries a ``created_at``/``ttl`` pair, and
``booking confirm`` / ``pairing confirm`` check it via
``StatefulProtocolSkill._expired`` and fail with exit 1 once the window has
passed. ``booking extend --resource <r> --ttl <n>`` is the legitimate escape
hatch, and it too refuses an already-expired hold.

Binding the plugin to those families turns the deadline from narrative colour
into something the effect stream can be graded on: the terminal
``booking_confirmed`` / ``pairing_completed`` effect exists only if the solver
actually beat the clock, because the skill itself rejects the late path.

Note that virtual time advances on ``wait``, so the deadline bites when the
trajectory genuinely spends time -- which is the realistic version of this
conflict (the agent is forced to choose between waiting for a slow party and
holding the window open) rather than a wall-clock gotcha.

Only seeds that expose ``booking`` or ``pairing`` pick the mechanic up; for the
others (``_mechanic_command_sets`` intersects with ``allowed_tool_families``)
the declaration is inert and the plugin stays narrative-only.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class TimeWindowExpirationPlugin(BaseConflictPlugin):
  """Plugin modeling temporal expiration windows and expired session states."""

  SPEC = ConflictPluginSpec(
      name="time_window_expiration",
      summary=(
          "A time-limited hold, invitation, or pairing session that expires "
          "unless synchronized coordination actions are completed before the "
          "window/SLA closes."
      ),
      category="timeout",
      skill_families=("booking", "pairing"),
      emits_effects=("booking_confirmed", "pairing_completed"),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "time_window_expiration"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Detection of an expiring resource, hold, or pairing session, "
        "and 2) Executing synchronized coordination actions before the session or SLA expires."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model a subagent with a time-limited capability: they must warn about the fast-approaching "
        "expiration window and reject requests if the solver does not provide inputs or synchronize actions in time."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    base = (
        "Create active temporary sessions, invitations, or configuration holds with explicit expiration "
        "timestamps (e.g. valid for only 300 seconds from start)."
    )
    if not self.mechanic_available(seed):
      # See DependencyOrderingPlugin.get_layer1b_instruction: narrative-only
      # seeds stay byte-identical to the pre-mechanic baseline.
      return base
    return base + (
        " State each window's length and start time explicitly, so the remaining budget is "
        "computable from the workspace rather than guessable."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    if not self.mechanic_available(seed):
      return ""
    return (
        "- EXPIRING WINDOW CHALLENGE: The trajectory opens the window (e.g. "
        "`booking hold --resource <target> --ttl <n>` or `pairing start "
        "--device <target> --ttl <n>`), completes the coordination steps, and "
        "closes it with the terminal confirm BEFORE the window lapses. If the "
        "trajectory must spend time waiting on another party, it either "
        "budgets the wait to stay inside the window or refreshes it first "
        "with `booking extend --resource <target> --ttl <n>`. The coordinator "
        "MUST NOT let the window close and then confirm: the skill rejects an "
        "expired hold or session with exit 1 and emits no terminal effect, "
        "which is precisely the missed-deadline failure this round grades "
        "against. Note `booking extend` also refuses an already-expired hold, "
        "so the refresh has to happen while the window is still open."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    if not self.mechanic_available(seed):
      return ""
    return (
        "- EVALS: Add an effect check asserting the coordinated action "
        "completed inside its window (`booking_confirmed` / "
        "`pairing_completed`) on the correct target -- i.e. the deadline was "
        "actually met, rather than the expiry being narrated and then missed."
    )
