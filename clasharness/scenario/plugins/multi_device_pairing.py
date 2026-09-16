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

"""Multi-device pairing handshake with an expiring session conflict plugin (M3, #9).

Steers the generator to build an ordered, short-lived device-pairing handshake
that must be driven to completion *in sequence* and *before its session window
closes*. A pairing session is opened for a device, a code is shown, the peer
enters the code, and only then can the pairing be confirmed — all within a single
TTL measured from when the session started. This is the ``multi_device_pairing``
mechanic; it pairs with the ``time_window_expiration`` framing on the same seed
(the session's expiry is *why* the handshake ordering and pace matter).

The trajectory/evals hooks push the generator to emit
``pairing start`` → ``pairing show-code`` → ``pairing enter-code`` →
``pairing confirm`` steps, advancing virtual time only with ``wait`` and staying
inside the window, which the checker scaffolder turns into a
``pairing_completed`` effect check. The tempting-but-wrong actions are letting
the session lapse before confirming, or skipping a step in the handshake — both
fail with exit 1 and emit no ``pairing_completed`` effect.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class MultiDevicePairingPlugin(BaseConflictPlugin):
  """Plugin modeling an ordered pairing handshake that must finish before expiry."""

  SPEC = ConflictPluginSpec(
      name="multi_device_pairing",
      summary=(
          "An ordered, short-lived device-pairing handshake (start -> show a "
          "code -> enter the code -> confirm) that must be completed in sequence "
          "before the session's time window closes; letting the session expire "
          "or skipping a step is the failure mode."
      ),
      category="timeout",
      skill_families=("pairing",),
      emits_effects=("pairing_completed",),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "multi_device_pairing"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery that linking the devices requires an "
        "ordered pairing handshake — open a session, reveal a one-time code, have "
        "the peer enter that code, and only then confirm — where the whole "
        "handshake is valid only for a short window and the session auto-expires "
        "if left incomplete; and 2) Driving the handshake to completion in the "
        "correct order before that window closes rather than skipping a step or "
        "letting the session lapse."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model a peer device or subagent that owns the pairing clock: it opens "
        "the session with an explicit TTL, expects each handshake step in order, "
        "and refuses to confirm once the session has expired or if a step was "
        "skipped. A stakeholder should press the coordinator to handle other "
        "work first, tempting it to let the pairing window lapse before "
        "confirming."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create workspace files showing the pairing session and its deadline: a "
        "session record with an explicit expiration (e.g. valid for only 300 "
        "seconds from when it started) and the required step order, plus runbook "
        "notes stating that the code must be shown and entered before confirming "
        "and that an expired session cannot be confirmed and must be restarted."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- ORDERED PAIRING DEADLINE CHALLENGE: The trajectory opens the session "
        "with `pairing start --device <d> --ttl <n>`, then walks the handshake "
        "in order — `pairing show-code --device <d>`, `pairing enter-code "
        "--device <d>`, `pairing confirm --device <d>` — advancing virtual time "
        "only with `wait <seconds>` when it must and finishing BEFORE the TTL "
        "elapses. The coordinator MUST NOT skip a step (confirming before the "
        "code is entered fails) and MUST NOT let the window lapse and then "
        "confirm: either case fails and emits no `pairing_completed` effect."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- EVALS: Add an effect check asserting the pairing was completed within "
        "its window (`pairing_completed`) and that the completion followed the "
        "ordered handshake on the same device, i.e. the code was shown and "
        "entered before confirming and the confirmation happened before the "
        "session's time window closed rather than after it had already expired."
    )
