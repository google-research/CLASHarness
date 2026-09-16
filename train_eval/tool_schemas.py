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

"""Native function/tool-calling schemas derived from AgentToolSpec (WS2).

This module turns the enriched ``agent_tools.v1.json`` command surface into
provider-native function/tool schemas, and reverse-maps a returned tool call
(``name`` + ``arguments``) back into the workspace CLI string that ``env.step``
already understands.

Design goals:

* **Behaviour parity with text-command eval.** The native-tool surface mirrors
  the existing text catalog exactly -- the ``safe_for_gt`` command subset (plus
  the ``done`` control tool) and the same per-task subsetting
  (:func:`train_eval.command_catalog._command_specs_for_task`). Every reverse-
  mapped call is a plain CLI string that flows through the *same*
  ``validate_command_against_spec`` + ``env.step`` path as text mode, so the two
  modes stay directly comparable.
* **Required-argument enforcement by the provider.** Each command's ``required``
  (derived in WS1 strictly from ``required_flags`` + ``required_positionals``)
  is surfaced as the function schema's ``required`` list, letting the provider
  refuse/repair a call that omits a mandatory argument -- eliminating the
  ``incomplete_required_submission`` failure mode.

The provider-agnostic :class:`ToolDef` is formatted for each vendor by the
``to_*_tools`` helpers. Provider *wiring* (request/response plumbing) lives in
the agent (WS3-5); this module is pure and independently unit-tested.
"""

from __future__ import annotations

import functools
import re
import shlex
from dataclasses import dataclass

from clasharness.core.task import Task
from clasharness.tools.agent_tool_spec import AgentCommandSpec, iter_command_specs

# The control tool that replaces the literal ``done`` text sentinel. Its CLI
# form is exactly ``done`` (special-cased by ``validate_command_against_spec``).
DONE_TOOL_NAME = "done"
_DONE_DESCRIPTION = (
    "Signal that the task is complete. Call this only after every required "
    "action and side effect has been performed."
)

_GEMINI_TYPES = {
    "string": "STRING",
    "boolean": "BOOLEAN",
    "integer": "INTEGER",
    "number": "NUMBER",
    "object": "OBJECT",
    "array": "ARRAY",
}


def tool_name_for_tokens(tokens: tuple[str, ...]) -> str:
    """Map CLI tokens to a provider-safe function name.

    ``("calendar", "add-event")`` -> ``"calendar_add-event"`` -> ``calendar_add_event``.
    Hyphens (which some providers disallow) are normalised to underscores. WS1
    verified this normalisation is collision-free across the safe command set.
    """
    return "_".join(tokens).replace("-", "_")


@dataclass(frozen=True)
class ToolDef:
    """A provider-agnostic tool definition.

    ``parameters`` is a plain JSON-Schema object with the WS1 ``x_cli`` reverse-
    map extension stripped out (providers must not see it).
    """

    name: str
    description: str
    parameters: dict
    tokens: tuple[str, ...]
    spec: AgentCommandSpec | None = None  # None only for the synthetic `done`.


# --------------------------------------------------------------------------- #
# Schema construction
# --------------------------------------------------------------------------- #
def _parameters_from_spec(spec: AgentCommandSpec) -> dict:
    """Build a provider-agnostic JSON-Schema object from an enriched spec.

    Copies only ``type``/``description``/``enum`` per property (dropping the
    internal ``x_cli`` extension) and carries over the ``required`` list.
    """
    args = spec.args or {}
    props_in = args.get("properties") or {}
    properties: dict[str, dict] = {}
    for pname, pdef in props_in.items():
        prop: dict = {"type": pdef.get("type", "string")}
        description = pdef.get("description")
        if description:
            prop["description"] = description
        if "enum" in pdef:
            prop["enum"] = list(pdef["enum"])
        properties[pname] = prop
    schema: dict = {"type": "object", "properties": properties}
    required = list(args.get("required") or [])
    if required:
        schema["required"] = required
    return schema


def _tool_def_from_spec(spec: AgentCommandSpec) -> ToolDef:
    return ToolDef(
        name=tool_name_for_tokens(spec.tokens),
        description=spec.description or spec.name,
        parameters=_parameters_from_spec(spec),
        tokens=spec.tokens,
        spec=spec,
    )


def _done_tool_def() -> ToolDef:
    return ToolDef(
        name=DONE_TOOL_NAME,
        description=_DONE_DESCRIPTION,
        parameters={"type": "object", "properties": {}},
        tokens=("done",),
        spec=None,
    )


@functools.lru_cache(maxsize=1)
def _safe_specs_by_tokens() -> dict[tuple[str, ...], AgentCommandSpec]:
    return {c.tokens: c for c in iter_command_specs(safe_for_gt=True)}


@functools.lru_cache(maxsize=1)
def _safe_specs_by_name() -> dict[str, AgentCommandSpec]:
    return {
        tool_name_for_tokens(c.tokens): c
        for c in iter_command_specs(safe_for_gt=True)
    }


def build_tool_defs(task: Task | None = None) -> list[ToolDef]:
    """Return the ordered tool defs for ``task`` (mirrors the text catalog subset).

    The ordering follows :func:`_command_specs_for_task` for determinism, with
    the ``done`` control tool appended last.
    """
    # Imported lazily to avoid a module import cycle (command_catalog imports
    # agent_tool_spec, this module also imports agent_tool_spec).
    from train_eval.command_catalog import _command_specs_for_task

    by_tokens = _safe_specs_by_tokens()
    tools: list[ToolDef] = []
    seen: set[str] = set()
    for local in _command_specs_for_task(task):
        spec = by_tokens.get(local.tokens)
        if spec is None:
            # e.g. a synthetic catalog entry with no rich spec; skipped because
            # it has no args schema / reverse-map. (None expected in practice.)
            continue
        name = tool_name_for_tokens(spec.tokens)
        if name in seen:
            continue
        seen.add(name)
        tools.append(_tool_def_from_spec(spec))
    tools.append(_done_tool_def())
    return tools


# --------------------------------------------------------------------------- #
# Reverse map: tool call -> workspace CLI string
# --------------------------------------------------------------------------- #
def _present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    return True


def _stringify(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        # Comma-list arguments (e.g. --attendees, --options) may arrive as arrays.
        return ",".join(_stringify(v) for v in value)
    return str(value)


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def command_from_tool_call(name: str, arguments: dict | None) -> str:
    """Render a provider tool call into a workspace CLI command string.

    Uses the WS1 ``x_cli`` extension to place positionals (in index order) and
    flags. Raises ``KeyError`` for an unknown tool name. Absent/empty optional
    arguments are omitted; the resulting string is intended to flow through the
    same ``validate_command_against_spec`` + ``env.step`` path as text mode.
    """
    arguments = arguments or {}
    if name == DONE_TOOL_NAME:
        return "done"
    spec = _safe_specs_by_name().get(name)
    if spec is None:
        raise KeyError(f"unknown tool: {name!r}")
    return _render_cli(spec, arguments)


def _keynorm(name: str) -> str:
    """Case/separator-insensitive form of an argument name."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonicalise_arguments(props: dict, arguments: dict) -> dict:
    """Map provider argument keys onto the spec's declared property names.

    Providers occasionally echo a property's *description* casing back as the
    argument key (``"Seconds"`` for ``seconds``, ``"TTL"`` for ``ttl``). Such a
    key is not a declared property, so :func:`_render_cli` used to drop it --
    silently, when the property was optional or a positional, producing a valid
    but under-specified command (e.g. ``agent add`` with no name).

    Exact matches always win. A non-matching key is retargeted only when its
    normalised form identifies a declared property that is still unfilled, so a
    caller that supplies both spellings keeps the exact one. A genuinely unknown
    key is still dropped, and the rendered command still fails
    ``validate_command_against_spec`` exactly as before.
    """
    out = {key: value for key, value in arguments.items() if key in props}
    by_norm: dict[str, str] = {}
    for pname in props:
        by_norm.setdefault(_keynorm(pname), pname)
    for key, value in arguments.items():
        if key in props:
            continue
        target = by_norm.get(_keynorm(key))
        if target is not None and target not in out:
            out[target] = value
    return out


def _render_cli(spec: AgentCommandSpec, arguments: dict) -> str:
    parts: list[str] = [" ".join(spec.tokens)]
    props = (spec.args or {}).get("properties") or {}
    arguments = _canonicalise_arguments(props, arguments)

    positionals = sorted(
        (pdef["x_cli"]["index"], pname)
        for pname, pdef in props.items()
        if pdef.get("x_cli", {}).get("kind") == "positional"
    )
    for _index, pname in positionals:
        if pname in arguments and _present(arguments[pname]):
            parts.append(shlex.quote(_stringify(arguments[pname])))

    for pname, pdef in props.items():
        x_cli = pdef.get("x_cli", {})
        kind = x_cli.get("kind")
        if kind == "flag":
            if pname in arguments and _present(arguments[pname]):
                parts.append(
                    f"{x_cli['flag']} {shlex.quote(_stringify(arguments[pname]))}"
                )
        elif kind == "flag_bool":
            if _truthy(arguments.get(pname)):
                parts.append(x_cli["flag"])

    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Provider formatters
# --------------------------------------------------------------------------- #
def to_openai_chat_tools(tools: list[ToolDef]) -> list[dict]:
    """OpenAI Chat Completions ``tools`` payload (nested ``function``)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def to_openai_responses_tools(tools: list[ToolDef]) -> list[dict]:
    """OpenAI Responses API ``tools`` payload (flat function tool)."""
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


def to_anthropic_tools(tools: list[ToolDef]) -> list[dict]:
    """Anthropic Messages API ``tools`` payload (``input_schema``)."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _to_gemini_schema(schema: dict) -> dict:
    """Convert a JSON-Schema object to the Gemini/OpenAPI ``Schema`` form."""
    out: dict = {}
    if "type" in schema:
        out["type"] = _GEMINI_TYPES.get(schema["type"], str(schema["type"]).upper())
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = list(schema["enum"])
    if "properties" in schema:
        out["properties"] = {
            k: _to_gemini_schema(v) for k, v in schema["properties"].items()
        }
    if "required" in schema:
        out["required"] = list(schema["required"])
    if "items" in schema:
        out["items"] = _to_gemini_schema(schema["items"])
    return out


def to_gemini_tools(tools: list[ToolDef], *, unified: bool = False) -> list[dict]:
    """Gemini ``tools`` payload: a single tool with ``functionDeclarations``.

    Two schema *dialects* are supported (the ``--llm-tool-schema`` ablation),
    both holding the tool *set* and semantics constant and varying only how each
    argument schema is expressed on the wire:

    * ``unified=False`` (native, default): each declaration carries a
      ``parameters`` block in Gemini's own OpenAPI ``Schema`` dialect (uppercase
      ``STRING``/``OBJECT`` type names via :func:`_to_gemini_schema`). This is
      Gemini's idiomatic function-declaration form.
    * ``unified=True``: each declaration carries a ``parameters_json_schema``
      block holding the *canonical* JSON-Schema verbatim -- byte-for-byte the
      same object OpenAI and Anthropic already receive -- so a cross-provider run
      differs only by the mandatory per-API envelope, not by schema dialect.
      Requires a google-genai SDK exposing
      ``FunctionDeclaration.parameters_json_schema`` (>= 1.x; verified on 2.x).

    OpenAI (chat + responses) and Anthropic already emit the canonical schema in
    both modes, so ``unified`` only changes Gemini -- the one provider whose
    native dialect actually diverges. The block is omitted for no-argument tools
    either way (Gemini rejects an empty ``OBJECT`` schema with no properties).
    """
    declarations: list[dict] = []
    for t in tools:
        decl: dict = {"name": t.name, "description": t.description}
        if (t.parameters.get("properties") or {}):
            if unified:
                # Canonical JSON-Schema, identical to the OpenAI/Anthropic wire
                # form; Gemini accepts it via the dedicated json-schema field.
                decl["parameters_json_schema"] = t.parameters
            else:
                decl["parameters"] = _to_gemini_schema(t.parameters)
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]
