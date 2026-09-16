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

from __future__ import annotations

import json
from pathlib import Path

import pytest

import examples.train_and_eval as train_and_eval
from clasharness.backend.multi_app_backend import MultiAppBackend
from clasharness.core.task import load_task, load_task_ids
from clasharness.tools.agent_tool_spec import (
    ToolSpecViolation,
    allowed_prefixes,
    iter_command_specs,
    load_agent_tool_spec,
    validate_command_against_spec,
    validate_task_solution_commands,
)
from clasharness.utils.safety_guard import ALLOWED_COMMAND_PREFIXES


def test_agent_tool_spec_loads_and_validates_core_commands():
    spec = load_agent_tool_spec()
    assert spec.version == "agent_tools.v1"
    assert {family.family for family in spec.families} >= {
        "file",
        "tasks",
        "email",
        "calendar",
        "weather",
        "system",
        "agent",
        "message",
        "plugin",
        "channel",
        "curl",
        "asset",
    }

    assert validate_command_against_spec("file list --path /").name == "file.list"
    assert validate_command_against_spec("asset read-text --id asset_1").name == "asset.read_text"
    assert (
        validate_command_against_spec("file create --path /x --content hello").name
        == "file.create"
    )
    assert validate_command_against_spec("done").name == "control.done"

    with pytest.raises(ToolSpecViolation, match="missing required flag"):
        validate_command_against_spec("file create --path /x")
    with pytest.raises(ToolSpecViolation, match="not defined"):
        validate_command_against_spec("unknown do")


def test_agent_tool_spec_rejects_duplicate_command_tokens(tmp_path: Path):
    spec_path = tmp_path / "agent_tools.v1.json"
    spec_path.write_text(
        json.dumps(
            {
                "version": "agent_tools.v1",
                "families": [
                    {
                        "family": "file",
                        "domains": ["file"],
                        "prefixes": ["file"],
                        "commands": [
                            {
                                "name": "file.one",
                                "tokens": ["file", "read"],
                                "syntax": "file read --path PATH",
                            },
                            {
                                "name": "file.two",
                                "tokens": ["file", "read"],
                                "syntax": "file read --path PATH",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ToolSpecViolation, match="duplicate command tokens"):
        load_agent_tool_spec(spec_path)


def test_train_and_eval_command_catalog_is_derived_from_safe_tool_spec():
    safe_tokens = {command.tokens for command in iter_command_specs(safe_for_gt=True)}
    catalog_tokens = {command.tokens for command in train_and_eval._COMMAND_SPECS}

    assert catalog_tokens == safe_tokens
    assert ("file", "list") in catalog_tokens
    assert ("asset", "read-text") in catalog_tokens
    assert ("curl",) not in catalog_tokens


def test_safety_allowlist_is_derived_from_tool_spec():
    assert tuple(ALLOWED_COMMAND_PREFIXES) == allowed_prefixes()
    assert "file " in ALLOWED_COMMAND_PREFIXES
    assert "system " in ALLOWED_COMMAND_PREFIXES


def test_tool_spec_prefixes_are_registered_by_multi_app_runtime():
    backend = MultiAppBackend()
    registered = set(backend._skill_registry.list_prefixes())
    routed_families = {
        "calendar",
        "email",
        "weather",
        "file",
        "asset",
        "tasks",
        "system",
        "agent",
        "message",
        "plugin",
        "channel",
        "curl",
    }

    expected = {
        prefix
        for family in load_agent_tool_spec().families
        if family.family in routed_families
        for prefix in family.prefixes
    }

    assert expected <= registered


def test_checked_in_ground_truth_commands_match_tool_spec():
    task_ids = load_task_ids("total")
    tasks = [load_task(task_id) for task_id in task_ids]

    assert task_ids
    assert validate_task_solution_commands(tasks) == []


def test_pro_file_only_family_rejects_non_file_gt():
    task = load_task(load_task_ids("total")[0])
    task.ground_truth.solution_commands = [
        "file create --path /reports/example.md --content ok",
        "email send --to a@example.com --subject hi --body body",
    ]

    errors = validate_task_solution_commands([task], allowed_families=["file"])
    assert errors
    assert "email send" in errors[0]


def test_wait_and_undo_are_always_allowed():
    task = load_task(load_task_ids("total")[0])
    task.ground_truth.solution_commands = [
        "file create --path /reports/example.md --content ok",
        "wait 5",
        "undo",
    ]
    # Should not raise errors for wait and undo, only for other families if they are present
    errors = validate_task_solution_commands([task], allowed_families=["file"])
    assert errors == []


def test_mcp_verb_commands_validate_against_spec():
    # The generator advertises (and the model emits) the MCP-tool verbs; these
    # must validate against their specs even though the runtime executes the
    # expanded ``python3``/``bash``/``unzip``/``tar``/``sqlite3`` forms.
    assert (
        validate_command_against_spec(
            "run_script --path scripts/reload_proxy.sh"
        ).name
        == "scripting.run"
    )
    assert (
        validate_command_against_spec(
            'run_script --path scripts/decrypt_key.py --args "--input a --output b"'
        ).name
        == "scripting.run"
    )
    assert (
        validate_command_against_spec(
            "decompress_archive --path bundle.zip --output-dir out"
        ).name
        == "archiving.decompress"
    )
    assert (
        validate_command_against_spec("decompress_archive --path logs.tar.gz").name
        == "archiving.decompress"
    )
    assert (
        validate_command_against_spec(
            'execute_db_query --db app.db --query "SELECT 1"'
        ).name
        == "database.query"
    )


def test_mcp_verb_run_script_requires_path_flag():
    # ``--path`` is a required flag per the advertised syntax; the bare positional
    # form (accepted only as a runtime convenience) is not a valid GT command.
    with pytest.raises(ToolSpecViolation, match="missing required flag"):
        validate_command_against_spec("run_script scripts/audit.py")


def test_mcp_verb_validation_respects_allowed_families():
    # run_script is rejected when its family is not in the task's allowlist, but
    # accepted when it is.
    with pytest.raises(ToolSpecViolation, match="not defined"):
        validate_command_against_spec(
            "run_script --path scripts/x.py", allowed_families=["file"]
        )
    assert (
        validate_command_against_spec(
            "run_script --path scripts/x.py",
            allowed_families=["shell_scripting"],
        ).name
        == "scripting.run"
    )


def test_shell_family_tokens_are_agent_verbs_prefixes_stay_runtime():
    families = {family.family: family for family in load_agent_tool_spec().families}
    scripting = families["shell_scripting"]
    # tokens (validation) are the agent-facing MCP verb...
    assert scripting.commands[0].tokens == ("run_script",)
    # ...while prefixes (safety allowlist / skill dispatch) stay the executables.
    assert scripting.prefixes == ("python3", "bash")
    # The runtime executables are therefore not what validation matches on.
    assert "run_script" not in scripting.prefixes


def test_agent_verbs_allow_first_token_outside_prefixes(tmp_path: Path):
    spec_path = tmp_path / "agent_tools.v1.json"
    spec_path.write_text(
        json.dumps(
            {
                "version": "agent_tools.v1",
                "families": [
                    {
                        "family": "shell_scripting",
                        "domains": ["scripting"],
                        "prefixes": ["python3", "bash"],
                        "agent_verbs": ["run_script"],
                        "commands": [
                            {
                                "name": "scripting.run",
                                "tokens": ["run_script"],
                                "syntax": "run_script --path PATH",
                                "required_flags": ["--path"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # tokens[0] == "run_script" is not in prefixes but is declared in agent_verbs.
    spec = load_agent_tool_spec(spec_path)
    assert spec.families[0].commands[0].tokens == ("run_script",)


def test_first_token_not_in_prefixes_or_agent_verbs_is_rejected(tmp_path: Path):
    spec_path = tmp_path / "agent_tools.v1.json"
    spec_path.write_text(
        json.dumps(
            {
                "version": "agent_tools.v1",
                "families": [
                    {
                        "family": "shell_scripting",
                        "domains": ["scripting"],
                        "prefixes": ["python3", "bash"],
                        "agent_verbs": ["run_script"],
                        "commands": [
                            {
                                "name": "scripting.run",
                                "tokens": ["perl"],
                                "syntax": "perl PATH",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ToolSpecViolation, match="is not in family prefixes"):
        load_agent_tool_spec(spec_path)


# --------------------------------------------------------------------------- #
# WS1: enriched `args` JSON schema (native-tool-calling migration).
# --------------------------------------------------------------------------- #
def _param_from_flag(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def test_every_shipped_command_has_enriched_args_schema():
    """Enrichment must be complete: every command exposes a properties schema."""
    for command in iter_command_specs():
        args = command.args
        assert "properties" in args, f"{command.name}: missing args.properties"
        assert isinstance(args["properties"], dict)
        for pname, pdef in args["properties"].items():
            assert "type" in pdef, f"{command.name}.{pname}: missing type"
            assert "description" in pdef and pdef["description"], (
                f"{command.name}.{pname}: missing description"
            )
            x_cli = pdef.get("x_cli")
            assert isinstance(x_cli, dict) and x_cli.get("kind") in (
                "flag",
                "flag_bool",
                "positional",
            ), f"{command.name}.{pname}: bad x_cli {x_cli!r}"


def test_enriched_args_required_matches_required_flags_and_positionals():
    """The schema's ``required`` is exactly the projection of ``required_flags`` +
    ``required_positionals`` onto its properties — the single source of truth the
    CLI validator and the (WS2) native-tool schema both derive from."""
    for command in iter_command_specs():
        props = command.args["properties"]
        required = set(command.args.get("required", []))

        flag_params = {
            p: d["x_cli"]["flag"]
            for p, d in props.items()
            if d["x_cli"]["kind"] in ("flag", "flag_bool")
        }
        positional_params = {
            p for p, d in props.items() if d["x_cli"]["kind"] == "positional"
        }

        # Every required flag maps to a required property.
        for flag in command.required_flags:
            matches = [p for p, f in flag_params.items() if f == flag]
            assert matches, f"{command.name}: required flag {flag} not in schema"
            assert matches[0] in required, (
                f"{command.name}: required flag {flag} not marked required"
            )

        # No *extra* required flags beyond required_flags.
        required_flag_params = {p for p in required if p not in positional_params}
        expected = {_param_from_flag(f) for f in command.required_flags}
        # Compare via the flag spelling to be robust to naming.
        got_flags = {flag_params[p] for p in required_flag_params}
        assert got_flags == set(command.required_flags), (
            f"{command.name}: required flags {got_flags} != "
            f"{set(command.required_flags)}"
        )
        assert required_flag_params == expected

        # Positional required count matches.
        got_pos = sum(1 for p in required if p in positional_params)
        assert got_pos == command.required_positionals, (
            f"{command.name}: required positionals {got_pos} != "
            f"{command.required_positionals}"
        )


def test_enriched_positional_indices_are_dense_and_ordered():
    """Positional x_cli indices are 0..k-1 with no gaps (needed for WS2 reverse-map)."""
    for command in iter_command_specs():
        indices = sorted(
            d["x_cli"]["index"]
            for d in command.args["properties"].values()
            if d["x_cli"]["kind"] == "positional"
        )
        assert indices == list(range(len(indices))), (
            f"{command.name}: non-dense positional indices {indices}"
        )


def _enriched_spec_doc(required_flags, required_positionals, properties, required):
    return {
        "version": "agent_tools.v1",
        "families": [
            {
                "family": "email",
                "domains": ["email"],
                "prefixes": ["email"],
                "commands": [
                    {
                        "name": "email.send",
                        "tokens": ["email", "send"],
                        "syntax": "email send --to EMAIL --subject SUBJECT",
                        "required_flags": required_flags,
                        "required_positionals": required_positionals,
                        "args": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    }
                ],
            }
        ],
    }


def test_enriched_args_reject_required_flag_not_marked_required(tmp_path: Path):
    spec_path = tmp_path / "agent_tools.v1.json"
    # --to is a required_flag but omitted from args.required -> inconsistent.
    spec_path.write_text(
        json.dumps(
            _enriched_spec_doc(
                required_flags=["--to"],
                required_positionals=0,
                properties={
                    "to": {"type": "string", "x_cli": {"kind": "flag", "flag": "--to"}},
                },
                required=[],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ToolSpecViolation, match="args.required flag params"):
        load_agent_tool_spec(spec_path)


def test_enriched_args_reject_required_flag_without_property(tmp_path: Path):
    spec_path = tmp_path / "agent_tools.v1.json"
    spec_path.write_text(
        json.dumps(
            _enriched_spec_doc(
                required_flags=["--subject"],
                required_positionals=0,
                properties={
                    "to": {"type": "string", "x_cli": {"kind": "flag", "flag": "--to"}},
                },
                required=[],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ToolSpecViolation, match="have no matching"):
        load_agent_tool_spec(spec_path)


def test_enriched_args_reject_positional_count_mismatch(tmp_path: Path):
    spec_path = tmp_path / "agent_tools.v1.json"
    spec_path.write_text(
        json.dumps(
            _enriched_spec_doc(
                required_flags=[],
                required_positionals=1,  # claims 1 required positional...
                properties={  # ...but schema has none required
                    "to": {"type": "string", "x_cli": {"kind": "flag", "flag": "--to"}},
                },
                required=[],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ToolSpecViolation, match="required positional count"):
        load_agent_tool_spec(spec_path)


def test_enriched_args_reject_unknown_x_cli_kind(tmp_path: Path):
    spec_path = tmp_path / "agent_tools.v1.json"
    spec_path.write_text(
        json.dumps(
            _enriched_spec_doc(
                required_flags=[],
                required_positionals=0,
                properties={
                    "to": {"type": "string", "x_cli": {"kind": "switch", "flag": "--to"}},
                },
                required=[],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ToolSpecViolation, match="unknown x_cli.kind"):
        load_agent_tool_spec(spec_path)

