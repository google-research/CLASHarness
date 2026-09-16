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

import json
import re
from typing import Any
from clasharness.scenario.spec import ScenarioSpec
from clasharness.tools.agent_tool_spec import normalize_command_quoting

# D-class: file extensions whose on-disk bytes are not the plaintext a
# ``file_check contains/equals <value>`` asserts. ``.enc`` (and other encrypted /
# compressed / binary blobs) hold ciphertext or binary; a ``*.template`` holds an
# unrendered placeholder, not the resolved value. A content assertion against
# such a file can never pass when replaying the reference trajectory, so it must
# be pruned rather than left to fail (or be papered over by a repair retry).
_OPAQUE_CONTENT_EXTENSIONS = frozenset({
    ".enc",
    ".gpg",
    ".age",
    ".gz",
    ".bz2",
    ".xz",
    ".zip",
    ".tar",
    ".7z",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".xlsx",
    ".xls",
    ".parquet",
    ".avro",
    ".proto.bin",
})

# Placeholder/templated files whose committed bytes are not the rendered output.
_TEMPLATE_CONTENT_SUFFIXES = (
    ".template",
    ".tmpl",
    ".tpl",
    ".j2",
    ".jinja",
    ".jinja2",
    ".in",
)


def _has_content_assertion(check: dict[str, Any]) -> bool:
  """True when a ``file_check`` asserts textual content (not mere existence)."""
  condition = check.get("condition")
  if condition in {"contains", "equals", "matches", "regex"}:
    return bool(str(check.get("expected", "")).strip())
  for key in ("contains", "equals", "matches", "regex", "value"):
    if str(check.get(key, "")).strip():
      return True
  return False


def _is_content_unassertable_path(path: str) -> bool:
  """True when a file's bytes cannot carry the asserted plaintext.

  Covers encrypted/compressed/binary blobs (``.enc`` etc.) and unrendered
  templates (``*.template``). Note this matches the *opaque file itself* - unlike
  Rule 6's ``encrypted_base_files`` set, which only holds the ``.enc``-stripped
  decrypted base name and therefore never catches a check aimed at the raw
  ``.enc`` path.
  """
  p = str(path).strip().lower().rstrip("/")
  if not p:
    return False
  if any(p.endswith(suffix) for suffix in _TEMPLATE_CONTENT_SUFFIXES):
    return True
  base = p.rsplit("/", 1)[-1]
  if ".proto.bin" in base and base.endswith(".proto.bin"):
    return True
  dot = base.rfind(".")
  if dot > 0 and base[dot:] in _OPAQUE_CONTENT_EXTENSIONS:
    return True
  return False


def audit_and_repair_scenario(spec: ScenarioSpec) -> ScenarioSpec:
    """Audit the ScenarioSpec rounds for contradictions and repair them in-place."""
    for round_item in spec.rounds:
        task_spec = round_item.task_spec
        if not isinstance(task_spec, dict):
            continue

        instruction = str(task_spec.get("instruction", ""))
        ground_truth = task_spec.get("ground_truth")
        if ground_truth is None or not isinstance(ground_truth, dict):
            ground_truth = {}

        checks = ground_truth.get("evaluation_checks") or []
        private_data = task_spec.get("data", {}).get("private", {})
        if not isinstance(private_data, dict):
            continue

        overrides = private_data.get("initial_state_overrides", {})
        if not isinstance(overrides, dict):
            continue

        # -------------------------------------------------------------
        # Rule 1: "No terminal commands" contradiction repair
        # -------------------------------------------------------------
        no_cmd_patterns = [
            r"do not attempt to run any terminal commands",
            r"do not run any terminal commands",
            r"do not run terminal commands",
            r"do not execute any terminal commands",
            r"do not attempt to run"
        ]
        is_no_cmd = any(re.search(pat, instruction, re.IGNORECASE) for pat in no_cmd_patterns)

        if is_no_cmd:
            # Find any file checklist or files_created checkers targeting the workspace
            for check in checks:
                target_path = ""
                contains_val = ""
                
                if check.get("type") == "file_check":
                    target_path = str(check.get("path", ""))
                    contains_val = str(check.get("contains", ""))
                elif check.get("effect_type") == "files_created":
                    expected = check.get("expected", {})
                    if isinstance(expected, dict) and expected.get("field") == "path":
                        target_path = str(expected.get("value", ""))
                        
                if target_path:
                    # We have a contradiction! The prompt forbids commands but expects a file check.
                    # Repair: Delegate the file modification/creation to a mock subagent completion
                    subagents = private_data.get("mock_subagents", [])
                    bg_events = private_data.get("background_events", [])
                    
                    # Target a subagent first, fallback to background events
                    targets = subagents if subagents else bg_events
                    if targets:
                        # Append the required file operation command to the state_effects of the first target
                        target = targets[0]
                        effects = target.setdefault("state_effects", [])
                        
                        # Strip 'workspace/' prefix if needed depending on tool execution
                        clean_path = target_path.replace("workspace/workspace/", "workspace/").replace("workspace/", "")
                        
                        # Generate content that satisfies the checker
                        content_str = contains_val if contains_val else "completed successfully"
                        
                        # Formulate append command
                        append_cmd = f"file append --path workspace/{clean_path} --content '\n\n- {content_str}: COMPLETED'"
                        
                        # Avoid duplicates
                        if append_cmd not in effects:
                            effects.append(append_cmd)

        # -------------------------------------------------------------
        # Rule 2: Checklist/Canary testing visible file dependency resolution
        # -------------------------------------------------------------
        # If any check requires canary testing details, ensure the corresponding file is visible
        has_canary_check = False
        for check in checks:
            expected_val = check.get("expected")
            if isinstance(expected_val, dict):
                criterion = str(expected_val.get("criterion", ""))
            else:
                criterion = ""
            if "canary" in criterion.lower() or "gate" in criterion.lower():
                has_canary_check = True
                break
                
        if has_canary_check:
            # Find the gate file in overrides
            gate_file_key = None
            for key in overrides:
                if "gate" in key.lower() or "canary" in key.lower():
                    gate_file_key = key
                    break
            
            if gate_file_key:
                # Strip the workspace/ prefix to get the filename/relative path
                ref_name = gate_file_key.replace("workspace/workspace/", "").replace("workspace/", "")
                
                # Add to visible workspace refs
                workspace_refs = round_item.workspace_refs
                if ref_name not in workspace_refs:
                    workspace_refs.append(ref_name)
                    
                # Also add to visible_refs in the round
                visible_refs = round_item.visible_refs
                ref_uri = f"workspace:{ref_name}"
                if ref_uri not in visible_refs:
                    visible_refs.append(ref_uri)
                    
                # Update task spec workspace refs if they exist
                task_spec["workspace_refs"] = workspace_refs

        # -------------------------------------------------------------
        # Rule 3 (REMOVED): anchor-ID injection into visible workspace files
        # -------------------------------------------------------------
        # Deliberately gone -- do not reintroduce. It appended a literal
        # "[System Note: Evidence anchor for this state is <ANCHOR>]" to the
        # first visible workspace file whenever an expected anchor was absent
        # from all of them. Its guard was `anchor not in combined_content` over
        # the visible text, so by construction it fired EXACTLY when the anchor
        # was carried only by a visual asset -- i.e. it targeted and erased the
        # asset-exclusivity this dataset exists to have. Measured on the two
        # generated sets it fired on 264/600 (44%) and 513/600 (85.5%) of
        # compiled tasks, naming an asset's own anchor in plain text and gluing
        # an obvious synthetic artifact onto author-written policy docs.
        #
        # The concern it nominally served is already owned, and owned better, by
        # llm_auditor.audit_scenario_with_llm ("Auditor 2: Anchor
        # Discoverability", audit_anchor_discoverability -> repair_missing_anchors),
        # which runs just BEFORE this layer. That auditor is handed
        # ``visible_visual_assets`` as well as the visible files, so it can
        # conclude an anchor is already discoverable through a visual asset and
        # report nothing missing; when it does repair it has the LLM weave the
        # anchor in naturally and skips binary/image targets. This rule saw only
        # the text, ignored the auditor's verdict, and re-injected unconditionally.

        # -------------------------------------------------------------
        # Rule 4: Flatten structured trigger/action dicts in mock_subagents
        # -------------------------------------------------------------
        PLACEHOLDERS = {
            "suggested_action",
            "target_component",
            "gate_status",
            "required_passcode",
            "workspace_file_provided",
        }

        def flatten_effect(eff: Any) -> list[str]:
            if not eff:
                return []
            if isinstance(eff, str):
                if eff not in PLACEHOLDERS:
                    return [eff]
                return []
            if isinstance(eff, list):
                res = []
                for child in eff:
                    res.extend(flatten_effect(child))
                return res
            if isinstance(eff, dict):
                # Handle nested subagent state_effects definition
                nested_effects = eff.get("state_effects") or eff.get("effects")
                if nested_effects and any(k in eff for k in ["agent_name", "agent_role", "agent"]):
                    return flatten_effect(nested_effects)

                # Handle trigger-action trigger patterns or list of actions
                actions = eff.get("actions") or eff.get("effects")
                if actions and isinstance(actions, list):
                    res = []
                    for action in actions:
                        res.extend(flatten_effect(action))
                    return res

                # Check for direct file write / command action
                action_type = eff.get("type") or eff.get("effect_type") or eff.get("action_type")
                if action_type:
                    action_type = str(action_type).lower()
                    if action_type in ("write_file", "file_write", "file_create", "create_file", "file_append", "append_file"):
                        path = eff.get("path", "")
                        content = eff.get("content", "")
                        import shlex
                        quoted_content = shlex.quote(content)
                        cmd_prefix = "file append" if "append" in action_type else "file create"
                        return [f"{cmd_prefix} --path {path} --content {quoted_content}"]
                    elif action_type in ("run_command", "command", "execute_command", "terminal_command"):
                        cmd = eff.get("command", "") or eff.get("cmd", "")
                        if cmd:
                            return [cmd]

                if "trigger" in eff and "effect" in eff:
                    return flatten_effect(eff.get("effect"))

            return []

        subagents = private_data.get("mock_subagents", [])
        for subagent in subagents:
            effects = subagent.get("state_effects", [])
            if not isinstance(effects, list):
                continue
            new_effects = []
            for effect in effects:
                new_effects.extend(flatten_effect(effect))
            # Repair invalid POSIX quoting (e.g. an apostrophe escaped as ``\'``
            # inside single quotes) in the flattened command strings -- including
            # those synthesized here from nested trigger/action dicts, which the
            # generation-time normalizer never sees.
            subagent["state_effects"] = [
                normalize_command_quoting(e) if isinstance(e, str) else e
                for e in new_effects
            ]

        # Defense-in-depth: normalize quoting in each round's reference
        # trajectory too. Generation already normalizes these, so this is a
        # no-op on valid commands, but it repairs any spec built outside the
        # generation loop (e.g. a reloaded legacy scenario).
        if isinstance(getattr(round_item, "expert_trajectory", None), list):
            round_item.expert_trajectory = [
                normalize_command_quoting(c) if isinstance(c, str) else c
                for c in round_item.expert_trajectory
            ]

        # -------------------------------------------------------------
        # Rule 5: Fix mismatched update IDs in rounds
        # -------------------------------------------------------------
        updates_in_spec = {u.update_id for u in spec.updates}
        for round_item in spec.rounds:
            round_id = round_item.round_id
            update_ids = round_item.update_ids
            if not isinstance(update_ids, list):
                continue
            new_update_ids = []
            for u_id in update_ids:
                if u_id in updates_in_spec:
                    new_update_ids.append(u_id)
                else:
                    # Search for mapped updates like u_r03_01 matching the round pattern
                    matched_id = None
                    for spec_u_id in updates_in_spec:
                        # Match if the round_id (e.g. 'r03') is present in the update_id (e.g. 'u_r03_01')
                        if round_id in spec_u_id:
                            matched_id = spec_u_id
                            break
                    if matched_id:
                        new_update_ids.append(matched_id)
                    else:
                        new_update_ids.append(u_id)
            round_item.update_ids = new_update_ids

        # -------------------------------------------------------------
        # Rule 6: Prune eval_item file_checks that can never hold on the raw
        # file bytes (opaque/encrypted/template content).
        #
        # Command-family pruning was removed here: commands are already validated
        # against the allowed families in-loop during generation (with retry), so
        # a post-hoc family filter is redundant. Worse, it keyed on
        # ``required_tool_families`` -- a field the compiled task_spec never
        # carries (it ships ``allowed_tool_families``) and round metadata never
        # sets -- so ``allowed_families`` was ALWAYS empty and the branch silently
        # dropped every recognized command_check from the (non-graded) eval_item
        # copy, without ever touching the task_spec copy that grading/F1 read.
        # -------------------------------------------------------------
        encrypted_base_files = set()
        if spec.workspace and hasattr(spec.workspace, "files") and spec.workspace.files:
            for f_dict in spec.workspace.files:
                if isinstance(f_dict, dict):
                    name = f_dict.get("path", "")
                    if name.endswith(".enc"):
                        encrypted_base_files.add(name.removesuffix(".enc"))


        for round_idx, round_item in enumerate(spec.rounds):
            round_id = round_item.round_id
            task_spec = round_item.task_spec or {}

            # D-class: prune opaque/template content assertions from the *shipped*
            # task_spec checks too. ``eval_item.checks`` (pruned below) and
            # ``task_spec.ground_truth.evaluation_checks`` are independent copies
            # after compilation (``task.to_dict()``); F1 replay and runtime grading
            # read the task_spec copy, so pruning only eval_items would leave the
            # doomed check shipping. Mutate the task_spec list in place.
            if isinstance(task_spec, dict):
                gt = task_spec.get("ground_truth")
                if isinstance(gt, dict) and isinstance(gt.get("evaluation_checks"), list):
                    gt["evaluation_checks"] = [
                        c
                        for c in gt["evaluation_checks"]
                        if not (
                            isinstance(c, dict)
                            and c.get("type") == "file_check"
                            and _is_content_unassertable_path(c.get("path", ""))
                            and _has_content_assertion(c)
                        )
                    ]

            for eval_item in spec.eval_items:
                if eval_item.round_id != round_id:
                    continue
                
                new_checks = []
                for check in eval_item.checks:
                    check_type = check.get("type")
                    if check_type == "file_check":
                        file_path = check.get("path", "")
                        # D-class: a content assertion against an encrypted/binary
                        # blob or an unrendered template can never hold on the raw
                        # file bytes (unlike the decrypted base name handled via
                        # ``encrypted_base_files`` below). Drop it; a bare
                        # existence check is left intact.
                        if _is_content_unassertable_path(file_path) and _has_content_assertion(check):
                            continue
                        if file_path in encrypted_base_files:
                            decrypted = False
                            for prev_idx in range(round_idx + 1):
                                prev_round = spec.rounds[prev_idx]
                                trajectory = " ".join(prev_round.expert_trajectory or [])
                                if "decrypt" in trajectory or "key" in trajectory:
                                    decrypted = True
                                    break
                            if not decrypted:
                                continue
                    new_checks.append(check)
                eval_item.checks = new_checks

    return spec


