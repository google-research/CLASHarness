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

"""Internal AgentToolSpec v1 loader and command validator."""

# pylint: disable=g-importing-member

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
import re
import shlex
import typing


class ToolSpecViolation(ValueError):  # pylint: disable=g-bad-exception-name
  """Raised when a command or tool spec violates AgentToolSpec."""


@dataclasses.dataclass(frozen=True)
class AgentCommandSpec:
  """One executable command surface exposed to agents."""

  family: str
  name: str
  tokens: tuple[str, ...]
  syntax: str
  domains: tuple[str, ...]
  args: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
  required_flags: tuple[str, ...] = ()
  required_positionals: int = 0
  state_effects: tuple[str, ...] = ()
  modes: tuple[str, ...] = ()
  safe_for_gt: bool = True
  description: str = ""
  note: str = ""


@dataclasses.dataclass(frozen=True)
class AgentToolFamily:
  """Command family in AgentToolSpec."""

  family: str
  domains: tuple[str, ...]
  prefixes: tuple[str, ...]
  commands: tuple[AgentCommandSpec, ...]


@dataclasses.dataclass(frozen=True)
class AgentToolSpec:
  """Parsed AgentToolSpec document."""

  version: str
  families: tuple[AgentToolFamily, ...]


def _default_spec_path() -> pathlib.Path:
  return (
      pathlib.Path(__file__).parents[1]
      / "specs"
      / "agent_tools.v1.json"
  )


@functools.lru_cache(maxsize=8)
def load_agent_tool_spec(
    path: str | pathlib.Path | None = None,
) -> AgentToolSpec:
  """Load and validate the internal AgentToolSpec JSON file."""

  spec_path = pathlib.Path(path) if path is not None else _default_spec_path()
  raw = json.loads(spec_path.read_text(encoding="utf-8"))
  version = str(raw.get("version", ""))
  if version != "agent_tools.v1":
    raise ToolSpecViolation(f"unsupported AgentToolSpec version: {version!r}")

  families: list[AgentToolFamily] = []
  seen_families: set[str] = set()
  seen_names: set[str] = set()
  seen_tokens: set[tuple[str, ...]] = set()
  for family_data in raw.get("families", []):
    family = str(family_data.get("family", "")).strip()
    if not family:
      raise ToolSpecViolation("tool family is missing `family`")
    if family in seen_families:
      raise ToolSpecViolation(f"duplicate tool family: {family}")
    seen_families.add(family)

    domains = tuple(str(item) for item in family_data.get("domains", []))
    prefixes = tuple(str(item) for item in family_data.get("prefixes", []))
    # ``agent_verbs`` are the agent-facing MCP-tool verbs (e.g. ``run_script``)
    # that the model emits and that AgentToolSpec validates. They intentionally
    # differ from ``prefixes`` (the runtime executables ``python3``/``bash``/...
    # used by the safety allowlist and skill dispatch) for the shell families:
    # ``rewrite_command`` expands the verb to its executable form only at
    # execution time. Kept out of ``prefixes`` so the safety allowlist and skill
    # registry continue to reflect what actually runs.
    agent_verbs = tuple(str(item) for item in family_data.get("agent_verbs", []))
    if not domains:
      raise ToolSpecViolation(f"family {family}: domains are required")
    if not prefixes:
      raise ToolSpecViolation(f"family {family}: prefixes are required")

    valid_first_tokens = set(prefixes) | set(agent_verbs)
    commands: list[AgentCommandSpec] = []
    for command_data in family_data.get("commands", []):
      command = _parse_command(family, domains, command_data)
      if command.name in seen_names:
        raise ToolSpecViolation(f"duplicate command name: {command.name}")
      if command.tokens in seen_tokens:
        raise ToolSpecViolation(
            f"duplicate command tokens: {' '.join(command.tokens)}"
        )
      if command.tokens[0] not in valid_first_tokens:
        raise ToolSpecViolation(
            f"command {command.name}: first token {command.tokens[0]!r} "
            f"is not in family prefixes {prefixes} or agent_verbs {agent_verbs}"
        )
      seen_names.add(command.name)
      seen_tokens.add(command.tokens)
      commands.append(command)

    if not commands:
      raise ToolSpecViolation(f"family {family}: commands are required")
    families.append(
        AgentToolFamily(
            family=family,
            domains=domains,
            prefixes=prefixes,
            commands=tuple(commands),
        )
    )

  if not families:
    raise ToolSpecViolation("AgentToolSpec must define at least one family")
  return AgentToolSpec(version=version, families=tuple(families))


def _validate_args_schema(
    name: str,
    args: dict[str, typing.Any],
    required_flags: tuple[str, ...],
    required_positionals: int,
) -> None:
  """Validate an *enriched* command ``args`` JSON schema for internal consistency.

  Only invoked when ``args`` declares ``"properties"`` (an enriched command);
  legacy/minimal ``args`` (empty or without a schema) are left untouched for
  backwards compatibility. Enforces the WS1 invariant that the schema's
  ``required`` list is exactly the projection of ``required_flags`` +
  ``required_positionals`` onto the schema's properties, so the native-tool
  schema (WS2) and the CLI validator can never disagree about what is required.
  """
  properties = args.get("properties")
  if not isinstance(properties, dict):
    raise ToolSpecViolation(f"command {name}: args.properties must be an object")
  required = args.get("required", [])
  if not isinstance(required, list):
    raise ToolSpecViolation(f"command {name}: args.required must be a list")

  flag_to_param: dict[str, str] = {}
  positional_params: set[str] = set()
  for pname, pdef in properties.items():
    if not isinstance(pdef, dict):
      raise ToolSpecViolation(
          f"command {name}: property {pname!r} must be an object"
      )
    x_cli = pdef.get("x_cli")
    if not isinstance(x_cli, dict) or "kind" not in x_cli:
      raise ToolSpecViolation(
          f"command {name}: property {pname!r} missing x_cli.kind"
      )
    kind = x_cli["kind"]
    if kind in ("flag", "flag_bool"):
      flag = x_cli.get("flag")
      if not isinstance(flag, str) or not flag.startswith("--"):
        raise ToolSpecViolation(
            f"command {name}: property {pname!r} has invalid x_cli.flag"
        )
      flag_to_param[flag] = pname
    elif kind == "positional":
      if not isinstance(x_cli.get("index"), int):
        raise ToolSpecViolation(
            f"command {name}: positional {pname!r} missing integer x_cli.index"
        )
      positional_params.add(pname)
    else:
      raise ToolSpecViolation(
          f"command {name}: property {pname!r} has unknown x_cli.kind {kind!r}"
      )

  required_set = set(required)
  if len(required_set) != len(required):
    raise ToolSpecViolation(f"command {name}: args.required has duplicates")
  for arg in required_set:
    if arg not in properties:
      raise ToolSpecViolation(
          f"command {name}: required arg {arg!r} is not a declared property"
      )

  # Every declared required flag must map to a property AND be marked required.
  missing_flags = [f for f in required_flags if f not in flag_to_param]
  if missing_flags:
    raise ToolSpecViolation(
        f"command {name}: required_flags {missing_flags} have no matching "
        "args property"
    )
  expected_required_flag_params = {flag_to_param[f] for f in required_flags}
  got_required_flag_params = {p for p in required_set if p not in positional_params}
  if got_required_flag_params != expected_required_flag_params:
    raise ToolSpecViolation(
        f"command {name}: args.required flag params {sorted(got_required_flag_params)} "
        f"!= required_flags projection {sorted(expected_required_flag_params)}"
    )
  got_required_positionals = sum(1 for p in required_set if p in positional_params)
  if got_required_positionals != required_positionals:
    raise ToolSpecViolation(
        f"command {name}: args.required positional count "
        f"{got_required_positionals} != required_positionals {required_positionals}"
    )


def _parse_command(
    family: str,
    family_domains: tuple[str, ...],
    data: dict[str, typing.Any],
) -> AgentCommandSpec:
  """Parse one command specification from specification dictionary."""
  name = str(data.get("name", "")).strip()
  tokens = tuple(str(item) for item in data.get("tokens", []))
  syntax = str(data.get("syntax", "")).strip()
  if not name:
    raise ToolSpecViolation(f"family {family}: command missing name")
  if not tokens:
    raise ToolSpecViolation(f"command {name}: tokens are required")
  if not syntax:
    raise ToolSpecViolation(f"command {name}: syntax is required")
  required_positionals = int(data.get("required_positionals", 0) or 0)
  if required_positionals < 0:
    raise ToolSpecViolation(
        f"command {name}: required_positionals must be >= 0"
    )
  args = data.get("args", {})
  if not isinstance(args, dict):
    raise ToolSpecViolation(f"command {name}: args must be an object")
  required_flags = tuple(str(item) for item in data.get("required_flags", []))
  if "properties" in args:
    _validate_args_schema(name, args, required_flags, required_positionals)
  return AgentCommandSpec(
      family=family,
      name=name,
      tokens=tokens,
      syntax=syntax,
      domains=tuple(str(item) for item in data.get("domains", family_domains)),
      args=dict(args),
      required_flags=required_flags,
      required_positionals=required_positionals,
      state_effects=tuple(str(item) for item in data.get("state_effects", [])),
      modes=tuple(str(item) for item in data.get("modes", [])),
      safe_for_gt=bool(data.get("safe_for_gt", True)),
      description=str(data.get("description", "")),
      note=str(data.get("note", "")),
  )


def iter_command_specs(
    *,
    safe_for_gt: bool | None = None,
    include_families: typing.Iterable[str] | None = None,
    spec: AgentToolSpec | None = None,
) -> tuple[AgentCommandSpec, ...]:
  """Return command specs, optionally filtered by GT safety and family."""

  resolved = spec or load_agent_tool_spec()
  families = set(include_families or [])
  commands: list[AgentCommandSpec] = []
  for family in resolved.families:
    if families and family.family not in families:
      continue
    for command in family.commands:
      if safe_for_gt is not None and command.safe_for_gt != safe_for_gt:
        continue
      commands.append(command)
  return tuple(commands)


def allowed_prefixes(*, include_inspection: bool = True) -> tuple[str, ...]:
  """Return safety allowlist prefixes derived from AgentToolSpec."""

  prefixes: list[str] = []
  for family in load_agent_tool_spec().families:
    if family.family == "shell_inspection" and not include_inspection:
      continue
    for prefix in family.prefixes:
      prefixes.append(
          prefix
          if prefix in {"pwd", "whoami", "env", "printenv"}
          else f"{prefix} "
      )
  return tuple(dict.fromkeys(prefixes))


def cli_verbs_by_family() -> dict[str, list[str]]:
  """Return simple two-token verbs for the random baseline agent."""

  verbs: dict[str, list[str]] = {}
  for command in iter_command_specs(safe_for_gt=True):
    if len(command.tokens) == 2 and command.family not in {"openclaw", "curl"}:
      verbs.setdefault(command.tokens[0], []).append(command.tokens[1])
  return {key: sorted(set(value)) for key, value in verbs.items()}


def required_flags_by_tokens() -> dict[tuple[str, ...], tuple[str, ...]]:
  """Return required flags for each command token key."""
  return {
      command.tokens: command.required_flags
      for command in iter_command_specs()
      if command.required_flags
  }


def required_positionals_by_tokens() -> dict[tuple[str, ...], int]:
  """Return required positional args counts for each command token key."""
  return {
      command.tokens: command.required_positionals
      for command in iter_command_specs()
      if command.required_positionals
  }


def normalize_family_name(name: str) -> str:
  """Normalize a tool family name to handle singular/plural and case mismatches."""
  name = (name or "").strip().lower()
  mapping = {
      "task": "tasks",
      "tasks": "tasks",
      "agent": "agent",
      "agents": "agent",
      "message": "message",
      "messages": "message",
      "file": "file",
      "files": "file",
      "email": "email",
      "emails": "email",
      "calendar": "calendar",
      "calendars": "calendar",
      "weather": "weather",
      "system": "system",
      "asset": "asset",
      "assets": "asset",
      "plugin": "plugin",
      "plugins": "plugin",
      "channel": "channel",
      "channels": "channel",
      "curl": "curl",
      "shell_inspection": "shell_inspection",
      "lock": "lock",
      "locks": "lock",
      "deploy": "deploy",
      "deploys": "deploy",
      "deployment": "deploy",
      "deployments": "deploy",
  }
  if name in mapping:
    return mapping[name]

  # Dynamic lookup fallback for custom families in the spec
  try:
    spec = load_agent_tool_spec()
    spec_families = {f.family.lower() for f in spec.families}
  except Exception:  # pylint: disable=broad-except
    spec_families = set()

  if name in spec_families:
    return name

  if name.endswith("ies") and name[:-3] + "y" in spec_families:
    return name[:-3] + "y"
  if name.endswith("ices"):
    if name[:-4] + "ex" in spec_families:
      return name[:-4] + "ex"
    if name[:-4] + "ix" in spec_families:
      return name[:-4] + "ix"
  if name.endswith("s") and name[:-1] in spec_families:
    return name[:-1]

  if name + "s" in spec_families:
    return name + "s"
  if (
      name.endswith("ex") or name.endswith("ix")
  ) and name[:-2] + "ices" in spec_families:
    return name[:-2] + "ices"
  if name.endswith("y") and name[:-1] + "ies" in spec_families:
    return name[:-1] + "ies"

  return name


# Free-text flags whose values routinely carry natural-language prose (and thus
# apostrophes/quotes). Used as a defensive fallback when a command does not match
# any known spec; the primary flag set is derived from the matched spec below.
_FALLBACK_FREETEXT_FLAGS: tuple[str, ...] = (
    "--message",
    "--instruction",
    "--body",
    "--subject",
    "--query",
    "--question",
    "--content",
    "--title",
    "--target",
    "--name",
    "--cron",
    "--options",
    "--role",
    "--location",
    "--channel",
    "--agent",
    "--to",
    "--id",
    "--path",
    "--src",
    "--dst",
    "--folder",
    "--model",
    "--args",
)


@functools.lru_cache(maxsize=1)
def _flags_by_command_prefix() -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
  """Return (command tokens, known flags) pairs, longest token prefix first."""

  pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
  try:
    specs = iter_command_specs()
  except Exception:  # pylint: disable=broad-except
    specs = ()
  for spec in specs:
    flags = set(spec.required_flags)
    flags.update(re.findall(r"--[a-zA-Z][\w-]*", spec.syntax or ""))
    pairs.append((spec.tokens, tuple(sorted(flags))))
  pairs.sort(key=lambda item: len(item[0]), reverse=True)
  return tuple(pairs)


def _recover_flag_value(raw_value: str) -> str:
  """Recover the intended literal from a possibly mis-quoted flag value token."""

  value = raw_value.strip()
  if value.startswith("="):
    value = value[1:].strip()
  if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
    value = value[1:-1]
  # Undo backslash-escaped quotes; ``\'`` inside POSIX single quotes is the
  # dominant generation defect (e.g. ``'...yesterday\'s...'``) that makes
  # ``shlex.split`` raise "No closing quotation".
  return value.replace("\\'", "'").replace('\\"', '"')


def normalize_command_quoting(command: str) -> str:
  """Best-effort re-quoting of a generated CLI command so it is POSIX-parseable.

  LLM-generated commands frequently wrap a free-text flag value (``--message``,
  ``--instruction``, ``--body`` ...) in single quotes while escaping an inner
  apostrophe as ``\\'`` -- which is invalid inside POSIX single quotes and makes
  ``shlex.split`` raise "No closing quotation". This normalizer detects the
  known flag values, recovers the intended literal, and re-emits them with
  ``shlex.quote``.

  Already-parseable commands are returned unchanged. If re-quoting cannot
  produce a parseable command, the original string is returned so the caller's
  validation still runs (this never makes a command worse).
  """

  raw = (command or "").strip()
  if not raw:
    return command
  try:
    shlex.split(raw)
    return command  # Already valid; leave untouched.
  except ValueError:
    pass

  # Resolve the known flags for this command from the longest matching spec.
  known_flags: tuple[str, ...] = _FALLBACK_FREETEXT_FLAGS
  for tokens, flags in _flags_by_command_prefix():
    prefix = " ".join(tokens)
    if flags and (raw == prefix or raw.startswith(prefix + " ")):
      known_flags = flags
      break

  # Locate the first occurrence of each known flag as a whitespace-delimited
  # token (also tolerating ``--flag=value``).
  positions: list[tuple[int, str]] = []
  for flag in known_flags:
    match = re.search(r"(?:^|\s)" + re.escape(flag) + r"(?=\s|=|$)", raw)
    if match:
      positions.append((match.end() - len(flag), flag))
  if not positions:
    return command
  positions.sort()

  head = raw[: positions[0][0]].rstrip()
  parts: list[str] = [head] if head else []
  for idx, (start, flag) in enumerate(positions):
    value_start = start + len(flag)
    value_end = positions[idx + 1][0] if idx + 1 < len(positions) else len(raw)
    value = _recover_flag_value(raw[value_start:value_end])
    parts.append(flag if value == "" else f"{flag} {shlex.quote(value)}")
  rebuilt = " ".join(part for part in parts if part)

  try:
    shlex.split(rebuilt)
  except ValueError:
    return command  # Give up; keep original so caller validation still runs.
  return rebuilt


def validate_command_against_spec(
    command: str,
    *,
    allowed_families: typing.Iterable[str] | None = None,
) -> AgentCommandSpec:
  """Validate a concrete CLI command against AgentToolSpec."""

  from clasharness.backend.openclaw_compat import rewrite_command  # pylint: disable=g-import-not-at-top

  raw = (command or "").strip()
  # Validate the agent-facing form: canonicalize legacy aliases but keep the
  # MCP-tool verbs (``run_script``/``decompress_archive``/``execute_db_query``)
  # intact so they match the ``scripting.run``/``archiving.decompress``/
  # ``database.query`` specs. Expansion to the runtime executables
  # (``python3``/``bash``/...) happens only at execution time (``env.step`` /
  # ``execute_cli``), which call ``rewrite_command`` with the default.
  decision = rewrite_command(raw, skip_incompatible=False, expand_mcp_verbs=False)
  raw = decision.executed_action.strip()
  if not raw:
    raise ToolSpecViolation("empty command")
  if raw.lower() == "done":
    return AgentCommandSpec(
        family="control",
        name="control.done",
        tokens=("done",),
        syntax="done",
        domains=("control",),
        safe_for_gt=True,
    )
  try:
    tokens = tuple(shlex.split(raw))
  except ValueError as exc:
    raise ToolSpecViolation(f"shell parse error: {exc}") from exc

  families = {normalize_family_name(f) for f in (allowed_families or [])}
  for spec in iter_command_specs():
    is_core_env_cmd = spec.name in ("system.wait", "system.undo")
    if not is_core_env_cmd and families and normalize_family_name(spec.family) not in families:
      continue
    if (
        len(tokens) < len(spec.tokens)
        or tokens[: len(spec.tokens)] != spec.tokens
    ):
      continue
    missing = [flag for flag in spec.required_flags if flag not in tokens]
    if missing:
      raise ToolSpecViolation(
          f"missing required flag(s) for {spec.name}: {', '.join(missing)}"
      )
    positional_tokens = [
        t for t in tokens[len(spec.tokens) :] if not t.startswith("--")
    ]
    if len(positional_tokens) < spec.required_positionals:
      raise ToolSpecViolation(
          f"missing required positional value(s) for {spec.name}"
      )
    return spec

  allowed = ", ".join(
      cmd.syntax for cmd in iter_command_specs(safe_for_gt=True)[:8]
  )
  raise ToolSpecViolation(
      f"command is not defined in AgentToolSpec: {raw}. Examples: {allowed}"
  )


def validate_task_solution_commands(
    tasks: typing.Iterable[typing.Any],
    *,
    allowed_families: typing.Iterable[str] | None = None,
) -> list[str]:
  """Return validation errors for task ground-truth command lists."""
  errors: list[str] = []
  for task in tasks:
    ground_truth = getattr(task, "ground_truth", None)
    commands = (
        getattr(ground_truth, "solution_commands", None)
        if ground_truth
        else None
    )
    if not commands:
      continue
    task_id = str(getattr(task, "task_id", "<unknown>"))
    for command in commands:
      try:
        validate_command_against_spec(
            str(command),
            allowed_families=allowed_families,
        )
      except ToolSpecViolation as exc:
        errors.append(f"{task_id}: {command} ({exc})")
  return errors
