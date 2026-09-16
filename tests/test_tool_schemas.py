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

"""Unit tests for the native tool-calling schema layer (WS2).

These tests are pure -- they exercise :mod:`train_eval.tool_schemas` against the
enriched ``agent_tools.v1.json`` surface without any provider I/O. The central
invariant is *round-trip parity*: every tool call the module emits must
reverse-map to a CLI string that the existing
``validate_command_against_spec`` + ``env.step`` path already accepts, so the
native-tool mode stays directly comparable to text-command mode.
"""

from __future__ import annotations

import pytest

import train_eval.tool_schemas as ts
from clasharness.core.task import GroundTruth, Task
from clasharness.tools.agent_tool_spec import (
    ToolSpecViolation,
    iter_command_specs,
    validate_command_against_spec,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dummy_arguments(spec) -> dict:
    """Fill every *required* property of ``spec`` with a schema-valid dummy."""
    args: dict = {}
    props = (spec.args or {}).get("properties") or {}
    for pname in (spec.args or {}).get("required") or []:
        pdef = props.get(pname, {})
        if "enum" in pdef:
            args[pname] = pdef["enum"][0]
        elif pdef.get("type") == "boolean":
            args[pname] = True
        else:
            args[pname] = "x"
    return args


def _messaging_task() -> Task:
    return Task(
        task_id="t_msg",
        instruction="send a message",
        domains=["messaging"],
        difficulty=1,
        initial_state="base",
        ground_truth=GroundTruth(
            solution_commands=["email send --to a@x.com --subject hi --body yo"],
            evaluation_checks=[],
        ),
    )


# --------------------------------------------------------------------------- #
# build_tool_defs
# --------------------------------------------------------------------------- #
def test_build_tool_defs_full_surface_has_done_last_and_no_duplicates():
    defs = ts.build_tool_defs(None)
    names = [d.name for d in defs]

    safe_count = len(iter_command_specs(safe_for_gt=True))
    # Every safe command + the synthetic ``done`` control tool, exactly once.
    assert len(defs) == safe_count + 1
    assert len(names) == len(set(names)), "duplicate tool names emitted"
    assert names[-1] == ts.DONE_TOOL_NAME
    assert names.count(ts.DONE_TOOL_NAME) == 1


def test_build_tool_defs_dedupes_synthetic_wait():
    # command_catalog appends a synthetic ``wait`` entry that shares tokens with
    # the ``system.wait`` spec; it must collapse to a single ``wait`` tool.
    names = [d.name for d in ts.build_tool_defs(None)]
    assert names.count("wait") == 1


def test_build_tool_defs_subsets_per_task():
    full = ts.build_tool_defs(None)
    sub = ts.build_tool_defs(_messaging_task())
    sub_names = {d.name for d in sub}

    assert 0 < len(sub) < len(full)
    assert "email_send" in sub_names  # from the task domain / ground truth
    assert ts.DONE_TOOL_NAME in sub_names  # done is always available


def test_done_tool_has_empty_object_schema_and_no_spec():
    done = ts.build_tool_defs(None)[-1]
    assert done.name == ts.DONE_TOOL_NAME
    assert done.spec is None
    assert done.parameters == {"type": "object", "properties": {}}


# --------------------------------------------------------------------------- #
# Parameters / schema hygiene
# --------------------------------------------------------------------------- #
def test_x_cli_never_leaks_into_provider_parameters():
    for d in ts.build_tool_defs(None):
        for prop in d.parameters.get("properties", {}).values():
            assert "x_cli" not in prop, f"x_cli leaked in {d.name}"


def test_required_list_is_surfaced_from_spec():
    add = next(d for d in ts.build_tool_defs(None) if d.name == "calendar_add_event")
    assert set(add.parameters.get("required", [])) == {"title", "start"}
    # Optional args are present in properties but not required.
    assert "location" in add.parameters["properties"]
    assert "location" not in add.parameters.get("required", [])


def test_enum_is_carried_into_parameters():
    mark = next(d for d in ts.build_tool_defs(None) if d.name == "email_mark")
    assert mark.parameters["properties"]["flag"]["enum"] == ["read", "unread"]


# --------------------------------------------------------------------------- #
# Reverse map: representative commands
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "arguments", "expected_cli", "expected_spec"),
    [
        # positional
        ("system_models_set", {"model": "gpt-4o"}, "system models set gpt-4o", "system.models_set"),
        # optional flags
        (
            "calendar_list",
            {"from": "2026-01-01", "to": "2026-02-01"},
            "calendar list --from 2026-01-01 --to 2026-02-01",
            "calendar.list",
        ),
        # enum flag
        ("email_mark", {"id": "abc", "flag": "read"}, "email mark --id abc --flag read", "email.mark"),
        # required flags
        (
            "email_send",
            {"to": "a@x.com", "subject": "hi", "body": "yo"},
            "email send --to a@x.com --subject hi --body yo",
            "email.send",
        ),
        # done sentinel
        ("done", {}, "done", "control.done"),
    ],
)
def test_reverse_map_representative_commands(name, arguments, expected_cli, expected_spec):
    cli = ts.command_from_tool_call(name, arguments)
    assert cli == expected_cli
    assert validate_command_against_spec(cli).name == expected_spec


def test_reverse_map_bool_flag_present_and_absent():
    assert ts.command_from_tool_call("system_status", {"json": True}) == "system status --json"
    # Falsy / absent boolean flags are omitted entirely.
    assert ts.command_from_tool_call("system_status", {"json": False}) == "system status"
    assert ts.command_from_tool_call("system_status", {}) == "system status"
    for cli in ("system status --json", "system status"):
        assert validate_command_against_spec(cli).name == "system.status"


def test_reverse_map_quotes_values_with_spaces():
    cli = ts.command_from_tool_call(
        "calendar_add_event", {"title": "Team Sync", "start": "2026-01-01T10:00"}
    )
    assert cli == "calendar add-event --title 'Team Sync' --start 2026-01-01T10:00"
    assert validate_command_against_spec(cli).name == "calendar.add_event"


def test_reverse_map_joins_list_valued_comma_argument():
    as_list = ts.command_from_tool_call(
        "calendar_add_event",
        {"title": "T", "start": "S", "attendees": ["a@x.com", "b@x.com"]},
    )
    as_string = ts.command_from_tool_call(
        "calendar_add_event",
        {"title": "T", "start": "S", "attendees": "a@x.com,b@x.com"},
    )
    assert "--attendees a@x.com,b@x.com" in as_list
    assert as_list == as_string
    assert validate_command_against_spec(as_list).name == "calendar.add_event"


def test_command_from_tool_call_unknown_name_raises():
    with pytest.raises(KeyError):
        ts.command_from_tool_call("definitely_not_a_tool", {})


def test_command_from_tool_call_tolerates_none_arguments():
    assert ts.command_from_tool_call("done", None) == "done"


# --------------------------------------------------------------------------- #
# Reverse map: full-surface round-trip invariant
# --------------------------------------------------------------------------- #
def test_every_tool_round_trips_through_validation():
    """Every emitted tool, given its required args, produces a valid CLI string."""
    for d in ts.build_tool_defs(None):
        if d.spec is None:  # done -> handled by the special case below.
            assert validate_command_against_spec(
                ts.command_from_tool_call(d.name, {})
            ).name == "control.done"
            continue
        cli = ts.command_from_tool_call(d.name, _dummy_arguments(d.spec))
        try:
            resolved = validate_command_against_spec(cli)
        except ToolSpecViolation as exc:  # pragma: no cover - failure detail
            pytest.fail(f"{d.name!r} -> {cli!r} failed validation: {exc}")
        assert resolved.name == d.spec.name


# --------------------------------------------------------------------------- #
# Reverse map: argument-key canonicalisation
# --------------------------------------------------------------------------- #
# Most property descriptions in agent_tools.v1.json are a case-echo of the key
# itself ("Seconds." for ``seconds``, "TTL." for ``ttl``), and providers do
# sometimes return the description's casing as the argument key. Such a key is
# not a declared property, so it used to be dropped by ``_render_cli`` -- loudly
# when the property was required, but *silently* when it was optional or a
# positional, yielding a well-formed but under-specified command.
def test_canonicalises_case_echoed_positional_key():
    # ``wait`` is the highest-volume instance: description "Seconds." -> key
    # "Seconds", dropped, leaving a bare ``wait`` that fails validation.
    cli = ts.command_from_tool_call("wait", {"Seconds": "5"})
    assert cli == "wait 5"
    assert validate_command_against_spec(cli).name == "system.wait"


def test_canonicalises_key_that_would_be_dropped_silently():
    # ``agent add`` has no required args, so a dropped name produced an accepted
    # command that registered an auto-named agent instead of the intended one.
    cli = ts.command_from_tool_call("agent_add", {"Name": "cohort-remap-runner"})
    assert cli == "agent add cohort-remap-runner"
    assert validate_command_against_spec(cli).name == "agent.add"


def test_canonicalises_all_caps_flag_key():
    cli = ts.command_from_tool_call(
        "approval_provisional",
        {"resource": "Production-A", "approver": "Silas", "TTL": "2h"},
    )
    assert cli == (
        "approval provisional --resource Production-A --approver Silas --ttl 2h"
    )
    assert validate_command_against_spec(cli).name == "approval.provisional"


def test_exact_key_wins_over_case_variant():
    # A caller supplying both spellings must keep the exactly-declared one; the
    # variant may only fill a slot that is still empty.
    cli = ts.command_from_tool_call(
        "pairing_confirm", {"device": "exact", "Device": "variant"}
    )
    assert cli == "pairing confirm --device exact"
    assert validate_command_against_spec(cli).name == "pairing.confirm"


def test_genuinely_unknown_key_is_still_dropped_and_rejected():
    # Canonicalisation must not turn an unknown argument into a silent success:
    # the key is still discarded and the command still fails validation.
    cli = ts.command_from_tool_call("file_read", {"totally_unknown": "/etc/passwd"})
    assert cli == "file read"
    with pytest.raises(ToolSpecViolation):
        validate_command_against_spec(cli)


def test_property_names_are_collision_free_under_canonicalisation():
    # Guards the rewrite rule itself: if a future command declared both ``foo``
    # and ``Foo`` (or ``foo-bar``/``foo_bar``), the alias mapping would be
    # ambiguous and could silently retarget an argument.
    for spec in iter_command_specs():
        props = list((spec.args or {}).get("properties") or {})
        normalised = [ts._keynorm(p) for p in props]
        assert len(set(normalised)) == len(normalised), (
            f"{spec.name} has property names that collide once normalised: {props}"
        )


# --------------------------------------------------------------------------- #
# Provider formatters
# --------------------------------------------------------------------------- #
def test_openai_chat_tools_shape():
    tool = ts.build_tool_defs(None)[0]
    payload = ts.to_openai_chat_tools([tool])[0]
    assert payload["type"] == "function"
    assert set(payload["function"]) == {"name", "description", "parameters"}
    assert payload["function"]["name"] == tool.name


def test_openai_responses_tools_shape_is_flat():
    tool = ts.build_tool_defs(None)[0]
    payload = ts.to_openai_responses_tools([tool])[0]
    assert payload["type"] == "function"
    assert payload["name"] == tool.name  # flat, not nested under "function"
    assert "function" not in payload


def test_anthropic_tools_use_input_schema():
    tool = ts.build_tool_defs(None)[0]
    payload = ts.to_anthropic_tools([tool])[0]
    assert set(payload) == {"name", "description", "input_schema"}
    assert payload["input_schema"] == tool.parameters


def test_gemini_tools_uppercase_types_and_omit_empty_params():
    defs = ts.build_tool_defs(None)
    gem = ts.to_gemini_tools(defs)
    assert list(gem[0]) == ["functionDeclarations"]
    decls = {d["name"]: d for d in gem[0]["functionDeclarations"]}

    # No-argument tool (done) omits the parameters block entirely.
    assert "parameters" not in decls[ts.DONE_TOOL_NAME]

    # Typed tool uses uppercase OpenAPI type names.
    mark = decls["email_mark"]
    assert mark["parameters"]["type"] == "OBJECT"
    assert mark["parameters"]["properties"]["flag"]["type"] == "STRING"
    assert mark["parameters"]["properties"]["flag"]["enum"] == ["read", "unread"]


def test_gemini_bool_type_maps_to_boolean():
    status = next(d for d in ts.build_tool_defs(None) if d.name == "system_status")
    decl = ts.to_gemini_tools([status])[0]["functionDeclarations"][0]
    assert decl["parameters"]["properties"]["json"]["type"] == "BOOLEAN"


def test_gemini_unified_uses_parameters_json_schema_verbatim():
    # In unified mode Gemini carries the canonical JSON-Schema (lowercase types)
    # verbatim under parameters_json_schema -- byte-for-byte the ToolDef schema
    # OpenAI/Anthropic receive -- and never the uppercase-OpenAPI ``parameters``.
    defs = ts.build_tool_defs(None)
    gem = ts.to_gemini_tools(defs, unified=True)
    assert list(gem[0]) == ["functionDeclarations"]
    decls = {d["name"]: d for d in gem[0]["functionDeclarations"]}
    by_name = {d.name: d for d in defs}

    mark = decls["email_mark"]
    assert "parameters" not in mark  # no native OpenAPI dialect
    assert mark["parameters_json_schema"] == by_name["email_mark"].parameters
    # Canonical JSON-Schema keeps lowercase types (contrast native "OBJECT").
    assert mark["parameters_json_schema"]["type"] == "object"
    assert (
        mark["parameters_json_schema"]["properties"]["flag"]["type"] == "string"
    )

    # No-argument tool (done) omits both schema blocks in unified mode too.
    assert "parameters" not in decls[ts.DONE_TOOL_NAME]
    assert "parameters_json_schema" not in decls[ts.DONE_TOOL_NAME]


def test_gemini_unified_matches_openai_anthropic_schema_content():
    # The whole point of unified mode: the argument-schema *content* is identical
    # across providers, so a cross-provider run varies only by API envelope.
    defs = ts.build_tool_defs(None)
    gem = {
        d["name"]: d
        for d in ts.to_gemini_tools(defs, unified=True)[0]["functionDeclarations"]
    }
    openai = {d["function"]["name"]: d for d in ts.to_openai_chat_tools(defs)}
    anthropic = {d["name"]: d for d in ts.to_anthropic_tools(defs)}

    name = "email_mark"
    canonical = openai[name]["function"]["parameters"]
    assert anthropic[name]["input_schema"] == canonical
    assert gem[name]["parameters_json_schema"] == canonical


def test_gemini_native_is_default_and_differs_from_unified():
    # Default (unified=False) stays the idiomatic uppercase-OpenAPI dialect, so
    # the flag is a real, opt-in variable rather than a silent behaviour change.
    defs = ts.build_tool_defs(None)
    native = ts.to_gemini_tools(defs)  # default
    assert native == ts.to_gemini_tools(defs, unified=False)
    mark = {d["name"]: d for d in native[0]["functionDeclarations"]}["email_mark"]
    assert mark["parameters"]["type"] == "OBJECT"
    assert "parameters_json_schema" not in mark


def test_gemini_unified_decls_are_accepted_by_genai_sdk():
    # Closes the loop with the real SDK: the unified declarations (canonical
    # JSON-Schema under parameters_json_schema) must construct a valid
    # FunctionDeclaration -- exactly how the agent builds them in
    # ``_gemini_tool_call``. Skipped where google-genai is unavailable.
    genai_types = pytest.importorskip("google.genai.types")
    assert "parameters_json_schema" in genai_types.FunctionDeclaration.model_fields, (
        "installed google-genai lacks FunctionDeclaration.parameters_json_schema;"
        " unified schema mode cannot target Gemini on this SDK"
    )
    defs = ts.build_tool_defs(None)
    decls = ts.to_gemini_tools(defs, unified=True)[0]["functionDeclarations"]
    built = [genai_types.FunctionDeclaration(**decl) for decl in decls]
    assert len(built) == len(defs)
    # A typed tool round-trips its canonical schema through the SDK model.
    mark = next(fd for fd in built if fd.name == "email_mark")
    assert mark.parameters_json_schema is not None
