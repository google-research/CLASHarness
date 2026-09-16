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

"""Effect checker - verifies side effects of agent actions."""

from __future__ import annotations

from typing import Any

from clasharness.evaluation.results import CheckResult
from clasharness.evaluation.evaluator import Evaluator



def _normalize_word(w: str) -> str:
    if w.endswith('ies') and len(w) > 3:
        return w[:-3] + 'y'
    if w.endswith('es') and len(w) > 2:
        return w[:-2]
    if w.endswith('s') and not w.endswith('ss') and len(w) > 1:
        return w[:-1]
    return w


def _tokenize_and_clean(text: str) -> list[str]:
    import re
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned.split()
    stop_words = {
        "the", "a", "an", "to", "for", "in", "on", "at", "by", "with", "of", "if",
        "their", "your", "my", "our", "his", "her", "its", "them", "you", "me", "us"
    }
    return [_normalize_word(w) for w in words if w not in stop_words]


def _is_subsequence(a: list[str], b: list[str]) -> bool:
    it = iter(b)
    return all(x in it for x in a)


def _format_transcript_line(idx: int, effect: dict[str, Any], field: str) -> str:
    """Render one effect as a judge-visible transcript line ("" = skip).

    Surfaces BOTH the addressed recipient and the channel the message was posted
    in, because criteria routinely assert the channel ("... notifies Priya in
    #access-requests-prod ...") -- 53% of the ``messages_sent`` semantic criteria
    in the conflict-VLM set name one explicitly.

    This used to collapse to ``target or recipient or channel``. Since
    ``message send`` requires ``--channel`` *and* ``--target``
    (``message_skill._action_send``), ``target`` is always present, so that
    chain short-circuited before ``channel`` and dropped it from every
    transcript the judge ever saw. The judge, asked to verify a routing fact
    that had been deleted from its input, answered things like "sent it directly
    rather than in the #billing-prod-migration channel" for messages that did
    carry exactly that ``--channel``.

    Restoring the channel is not a relaxation: it also lets the judge correctly
    REJECT a message posted to the wrong channel, which it previously could not
    see either.
    """
    val = str(effect.get(field, "")).strip()
    if not val:
        return ""

    recipient = str(effect.get("target") or effect.get("recipient") or "").strip()
    channel = str(effect.get("channel") or "").strip()
    sender = str(effect.get("sender") or "").strip()

    # ``message broadcast`` (message_skill) and ``_record_subagent_message``
    # (environment) both fall back to ``target = channel`` when there is no
    # distinct addressee. Render the location once, not "to #ops in #ops".
    if recipient and channel and recipient.lstrip("#").lower() == channel.lstrip("#").lower():
        recipient = ""
    if channel and not channel.startswith("#"):
        channel = f"#{channel}"

    if recipient and channel:
        where = f"to {recipient} in {channel}"
    elif recipient:
        where = f"to {recipient}"
    elif channel:
        where = f"in {channel}"
    else:
        where = ""

    # Surface the author when present so criteria phrased about a specific
    # speaker ("<Subagent> acknowledges ...") can be graded by attribution, not
    # content-guessing. Solver-sent messages carry no sender.
    if sender:
        prefix = f"From {sender} {where}: " if where else f"From {sender}: "
    elif where:
        prefix = f"{where[0].upper()}{where[1:]}: "
    else:
        prefix = ""
    return f"[{idx}] {prefix}{val}"


def _effect_field_contains(
    effect: dict[str, Any], field: str, value: Any, effect_type: str
) -> bool:
    """Whether ``effect[field]`` contains ``value`` (mirrors ``field_contains``).

    Tokenized, order-tolerant, stop-word-stripped subsequence match for
    ``messages_sent`` (as in the ``field_contains`` branch); plain
    case-insensitive substring otherwise. Used by the ``within_window`` content
    filter so an on-time assertion can be scoped to the *decisive* effect rather
    than any effect of the type. Kept additive: the ``field_contains`` branch is
    intentionally left untouched.
    """
    hay = str(effect.get(field, ""))
    needle = str(value).lower()
    if effect_type == "messages_sent":
        return _is_subsequence(_tokenize_and_clean(needle), _tokenize_and_clean(hay))
    return needle in hay.lower()


def _norm_scalar(value: Any) -> str:
    """Fold a payload value / expected identifier for equality comparison.

    Only scalars are folded; dicts and lists return "" so a nested structure can
    never accidentally equal an identifier. Comparison is
    case-insensitive and whitespace-trimmed because the command line and the
    recorded payload agree on the identifier's spelling but not reliably on its
    case (``DEV-1`` vs ``dev-1``).
    """
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    if isinstance(value, bool):
        return str(value).strip().lower()
    return str(value).strip().lower()


def _normalize_asset_base(asset_id: str) -> str:
    """Strip auto-generated timestamp/round suffixes and a trailing '_chart'.

    Lets a needle like ``api_gateway_chart`` match a generated asset id such as
    ``api_gateway_20260330_103000`` without per-scenario special-casing. Also
    folds a doubled platform prefix / path separators (e.g.
    ``workspace_workspace/flight_schedule`` or ``workspace/flight_schedule``)
    that older data may carry so it still matches the clean, single-prefixed
    ``workspace_flight_schedule`` id the generator now emits. Kept in sync with
    ``conflict_vlm_materializer._asset_base_key``.
    """
    import re
    asset_id = re.sub(r"_\d{8}_\d{6}", "", asset_id)
    asset_id = re.sub(r"_r\d+", "", asset_id)
    asset_id = re.sub(r"^([^/_]+)_\1/", r"\1_", asset_id)  # workspace_workspace/ -> workspace_
    asset_id = asset_id.replace("/", "_")  # any residual path separator -> '_'
    if asset_id.endswith("_chart"):
        asset_id = asset_id[:-6]
    return asset_id


class EffectChecker(Evaluator):
    """Checks that expected side effects occurred.

    Verifies things like: message was sent, cron job was created,
    plugin was installed, etc.
    """

    def __init__(
        self,
        effect_type: str,
        condition: str,
        expected: Any,
        name: str = "",
        weight: float = 1.0,
    ) -> None:
        self.effect_type = effect_type  # "message_sent", "cron_created", etc.
        self.condition = condition  # "exists", "count", "field_equals"
        self.expected = expected
        self.name = name or f"effect:{effect_type} {condition}"
        self.weight = weight

    def evaluate(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> CheckResult:
        effects = env_state.get("effects", {})
        effect_list = effects.get(self.effect_type, [])

        # Judge-infra status for the semantic branch; every deterministic branch
        # leaves these at their "ran fine" defaults.
        ok_flag = True
        err = ""
        # Only the semantic branch can leave this False (nothing to grade).
        judge_invoked = True

        if self.condition == "exists":
            passed = len(effect_list) > 0
            msg = f"{self.effect_type} occurred: {passed}"

        elif self.condition == "count":
            passed = len(effect_list) == self.expected
            msg = f"{self.effect_type} count: {len(effect_list)} == {self.expected}: {passed}"

        elif self.condition == "count_equals":
            passed = len(effect_list) == self.expected
            msg = f"{self.effect_type} count: {len(effect_list)} == {self.expected}: {passed}"

        elif self.condition == "count_gte":
            passed = len(effect_list) >= self.expected
            msg = f"{self.effect_type} count: {len(effect_list)} >= {self.expected}: {passed}"

        elif self.condition == "count_lte":
            passed = len(effect_list) <= self.expected
            msg = f"{self.effect_type} count: {len(effect_list)} <= {self.expected}: {passed}"

        elif self.condition == "field_equals":
            # expected is {"field": "target", "value": "+1234567890"}
            field = self.expected.get("field", "")
            value = self.expected.get("value", "")
            passed = any(
                e.get(field) == value for e in effect_list
            )
            msg = f"{self.effect_type}.{field} == '{value}': {passed}"

        elif self.condition == "field_contains":
            field = self.expected.get("field", "")
            value = self.expected.get("value", "")
            needle = str(value).lower()
            if self.effect_type == "messages_sent":
                needle_tokens = _tokenize_and_clean(needle)
                passed = any(
                    _is_subsequence(needle_tokens, _tokenize_and_clean(str(e.get(field, ""))))
                    for e in effect_list
                )
                msg = f"{self.effect_type}.{field} tokenized contains '{value}': {passed}"
            elif self.effect_type == "assets_accessed":
                vlm_anchors = task_data.get("private", {}).get("vlm_asset_anchors", {})
                manifest = task_data.get("private", {}).get("initial_state_overrides", {}).get("asset_manifest.json", {})
                assets_list = manifest.get("assets", []) if isinstance(manifest, dict) else []
                asset_paths = {str(a.get("id")).lower(): str(a.get("path")).lower() for a in assets_list if a.get("id") and a.get("path")}
                needle_path = asset_paths.get(needle)

                passed = False
                for e in effect_list:
                    asset_id = str(e.get("id")).lower() if e.get("id") else ""
                    if needle == asset_id:
                        passed = True
                        break
                    if needle_path and asset_paths.get(asset_id) == needle_path:
                        passed = True
                        break
                    # Platform ID container matching fallback
                    expected_asset_entry = next((a for a in assets_list if str(a.get("id")).lower() == needle), None)
                    if expected_asset_entry:
                        platform_id = str(expected_asset_entry.get("platform_id", "")).lower()
                        if platform_id and platform_id == asset_id:
                            passed = True
                            break
                    anchors_in_effect = [str(aid).lower() for aid in e.get("anchor_ids", [])]
                    if needle in anchors_in_effect:
                        passed = True
                        break
                    anchors_in_private = [str(aid).lower() for aid in vlm_anchors.get(asset_id, [])]
                    if needle in anchors_in_private:
                        passed = True
                        break

                # Generic normalization: match chart/asset IDs regardless of an
                # auto-generated timestamp suffix, round suffix, or trailing
                # "_chart". This is a scenario-agnostic convenience (agents are
                # not expected to reproduce exact generated timestamps), NOT a
                # per-scenario override.
                #
                # BUT it must never be allowed to erase the temporal_staleness
                # distinction, which is a core conflict dimension of this
                # benchmark: a scenario deliberately keeps older snapshots of
                # the same asset visible alongside the decisive one, and the
                # task is precisely to ground on the right one. Since
                # _normalize_asset_base() strips the timestamp,
                # "<asset>_20241024_081523" and a stale "<asset>" collapse to
                # the same base, so an unguarded fallback passes an agent that
                # read the WRONG snapshot. Two guards:
                #
                #   1. If the needle is itself a concrete visible asset id, the
                #      spec has already named exactly what must be read, so
                #      normalization can only ever loosen it incorrectly.
                #   2. Otherwise the normalized base must pick out exactly ONE
                #      visible asset. If several collapse to it, the fallback
                #      cannot tell which one the task meant and must not guess.
                #
                # Refusals are reported in ``msg`` so a failing check is
                # diagnosable rather than silently strict.
                fallback_note = ""
                if not passed:
                    candidate_ids = {
                        str(a.get("id")).lower() for a in assets_list if a.get("id")
                    }
                    # visible_asset_ids is a superset of the manifest (the
                    # manifest omits assets that are visible but uncatalogued),
                    # and ambiguity is decided by what the agent could reach.
                    candidate_ids |= {
                        str(x).lower()
                        for x in (task_data.get("private", {}).get("visible_asset_ids") or [])
                    }
                    needle_base = _normalize_asset_base(needle)
                    colliding = sorted(
                        a for a in candidate_ids if _normalize_asset_base(a) == needle_base
                    )
                    if needle in candidate_ids:
                        fallback_note = (
                            f"; normalized fallback disabled ('{needle}' is a concrete "
                            f"visible asset id -- exact match required)"
                        )
                    elif len(colliding) > 1:
                        fallback_note = (
                            f"; normalized fallback refused (base '{needle_base}' is "
                            f"ambiguous across {colliding})"
                        )
                    else:
                        for e in effect_list:
                            asset_id = str(e.get("id")).lower() if e.get("id") else ""
                            if needle_base == _normalize_asset_base(asset_id):
                                passed = True
                                break

                msg = (
                    f"{self.effect_type}.{field} contains '{value}': {passed}"
                    f"{'' if passed else fallback_note}"
                )
            else:
                passed = any(
                    needle in str(e.get(field, "")).lower() for e in effect_list
                )
                msg = f"{self.effect_type}.{field} contains '{value}': {passed}"

        elif self.condition == "field_semantic_match":
            from clasharness.evaluation.llm_judge import judge_majority as _llm_judge
            field = self.expected.get("field", "")
            criterion = self.expected.get("criterion", "")
            model = self.expected.get("model")

            transcript_items = []
            for idx, e in enumerate(effect_list, 1):
                line = _format_transcript_line(idx, e, field)
                if line:
                    transcript_items.append(line)

            if not transcript_items:
                # A positive semantic assertion cannot be satisfied by an empty
                # transcript. Fail closed WITHOUT calling the judge: grading
                # "(no items found)" wastes a model call and risks a spurious
                # pass hallucinated from the placeholder text.
                passed = False
                judge_invoked = False
                msg = (
                    f"LLM semantic match: no '{field}' content in "
                    f"{self.effect_type} to evaluate (empty transcript)"
                )
            else:
                transcript_text = "\n".join(transcript_items)
                # k-way self-consistency vote to smooth out borderline
                # single-shot flakiness in semantic grading.
                result = _llm_judge(transcript_text, criterion, model=model)
                passed = result.passed
                ok_flag = result.ok
                err = result.error
                msg = f"LLM semantic match: {result.reason or result.error}"

        elif self.condition == "within_window":
            # Temporal operator (M8/CL2, F5/F10). Passes iff at least one effect
            # of this type carries a ``virtual_time`` stamp inside the window
            # ``[min_virtual_time, max_virtual_time]`` (virtual seconds).
            # ``max_virtual_time`` is required (the SLA deadline);
            # ``min_virtual_time`` is an OPTIONAL lower bound (M8/CL2 L1). An
            # optional ``field``/``value`` content filter (same matching as
            # ``field_contains``) scopes the check to the *decisive* effect, so a
            # throwaway early effect cannot satisfy an on-time assertion about a
            # specific response (e.g. "answered the P1 ticket before its SLA
            # window closed").
            #
            # The lower bound closes a no-wait bypass: in a ``force_async_wait``
            # round the decisive sub-agent's reply legitimately arrives only when
            # its delayed completion fires, stamped at the sub-agent's own
            # ``trigger_time``. A no-wait solver that instead elicits the reply
            # synchronously (Route 2 mirror) has it stamped at ``virtual_time=0``,
            # which ``min_virtual_time`` rejects -- so ``wait`` becomes load-
            # bearing. Fail-closed: an absent, unstamped, past-deadline, or
            # before-the-window (too-early) effect cannot prove on-time
            # completion. ``virtual_time`` is stamped on effect dicts at emission
            # time by the environment (see ``CLASHarnessEnv._stamp_new_effects``).
            exp = self.expected if isinstance(self.expected, dict) else {}
            deadline = exp.get("max_virtual_time")
            min_vt = exp.get("min_virtual_time")
            field = exp.get("field", "")
            value = exp.get("value", "")
            has_filter = bool(field) and value != ""
            if deadline is None:
                passed = False
                msg = "within_window: missing 'max_virtual_time' in expected"
            else:
                candidates = effect_list
                if has_filter:
                    candidates = [
                        e for e in effect_list
                        if _effect_field_contains(e, field, value, self.effect_type)
                    ]
                on_time = [
                    e for e in candidates
                    if isinstance(e.get("virtual_time"), (int, float))
                    and not isinstance(e.get("virtual_time"), bool)
                    and e.get("virtual_time") <= deadline
                    and (min_vt is None or e.get("virtual_time") >= min_vt)
                ]
                passed = len(on_time) > 0
                filt = f" [{field} contains '{value}']" if has_filter else ""
                lo = "" if min_vt is None else f"{min_vt} <= "
                msg = (
                    f"{self.effect_type}{filt} within window "
                    f"({lo}virtual_time <= {deadline}): {passed}"
                )

        elif self.condition == "not_exists":
            passed = len(effect_list) == 0
            msg = f"{self.effect_type} did not occur: {passed}"

        elif self.condition == "any_field_equals":
            # expected is {"value": "DEV-4471"} -- "an effect of this type was
            # emitted FOR THIS TARGET", without naming the payload field.
            #
            # Field-agnostic on purpose. The identifier a protocol command is
            # invoked with is always recoverable from the command line (it is the
            # command's first required flag), but the payload key it lands under
            # is not: ``containment isolate-device --device D`` records it as
            # ``target``, ``lock acquire --resource R`` as ``resource``. Matching
            # on the value keeps the check derivable from the SSOT instead of
            # requiring a hand-maintained flag->field map that would silently rot.
            #
            # This is what makes a protocol check gradable rather than decorative.
            # A bare ``exists`` passes for ANY invocation, so a solver that runs
            # the right verb against the WRONG target still scores -- and choosing
            # the right target is the difficulty the conflict is built around. It
            # also removes the need to count: N commands against N distinct
            # targets produce N checks with N distinct expected values, so
            # repeating one target satisfies exactly one of them.
            want = _norm_scalar(
                self.expected.get("value") if isinstance(self.expected, dict)
                else self.expected
            )
            passed = False
            if want:
                for e in effect_list:
                    values = e.values() if isinstance(e, dict) else [e]
                    if any(_norm_scalar(v) == want for v in values):
                        passed = True
                        break
            msg = f"{self.effect_type} emitted for target '{want}': {passed}"

        else:
            passed = False
            msg = f"Unknown condition: {self.condition}"

        return CheckResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            weight=self.weight,
            message=msg,
            ok=ok_flag,
            error=err,
            judge_invoked=judge_invoked,
        )

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> EffectChecker:
        return cls(
            effect_type=spec["effect_type"],
            condition=spec.get("condition", "exists"),
            expected=spec.get("expected"),
            name=spec.get("name", ""),
            weight=spec.get("weight", 1.0),
        )


def _call_gemini_judge(content: str, criterion: str, model: str) -> tuple[bool, str]:
    """Backward-compatible shim delegating to the unified LLM judge.

    Returns ``(passed, reason_or_error)``. Unlike the previous implementation it
    never raises: a missing/failed judge yields ``(False, <error>)`` so a single
    misconfigured judge does not crash the whole evaluation.
    """
    from clasharness.evaluation.llm_judge import judge as _llm_judge

    result = _llm_judge(content, criterion, model=model)
    return result.passed, (result.reason or result.error)
