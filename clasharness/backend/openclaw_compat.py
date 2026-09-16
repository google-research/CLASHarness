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

"""Compatibility helpers for running legacy OpenClaw commands."""

from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass, field


@dataclass
class CompatDecision:
    """Compatibility decision for a single command."""

    original_action: str
    executed_action: str
    compat_status: str = "ok"  # ok | rewritten | skipped_incompatible
    error_tags: list[str] = field(default_factory=list)
    skip_reason: str = ""


def rewrite_command(
    command: str,
    skip_incompatible: bool = True,
    *,
    expand_mcp_verbs: bool = True,
) -> CompatDecision:
    """Rewrite legacy OpenClaw commands to improve runtime compatibility.

    ``expand_mcp_verbs`` controls the MCP-tool-verb -> executable expansion
    (``run_script``/``decompress_archive``/``execute_db_query`` ->
    ``python3``/``bash``/``unzip``/``tar``/``sqlite3``). It defaults to ``True``
    for the *execution* callers (``env.step`` / ``execute_cli``), which must run
    the underlying CLI. The *validation* caller
    (``validate_command_against_spec``) passes ``False`` so the agent-facing MCP
    verb is preserved and matched against the ``scripting.run`` /
    ``archiving.decompress`` / ``database.query`` specs (whose ``tokens`` are the
    MCP verbs). Keeping a single rewrite layer means the verb the model emits and
    the command that ultimately runs stay in lockstep.
    """
    decision = CompatDecision(original_action=command, executed_action=command)

    # MCP-tool verb -> executable CLI. The generator advertises these commands to
    # the model by their MCP tool *syntax* (``run_script``, ``decompress_archive``,
    # ``execute_db_query``; see AgentToolSpec ``syntax`` fields); the environment
    # executes the underlying CLI form (``python3``/``bash``, ``unzip``/``tar``,
    # ``sqlite3``). Expand only for execution callers - validation matches the
    # unexpanded verb - so the model's faithfully-followed syntax is accepted and
    # runs identically at eval.
    if expand_mcp_verbs:
        mcp_cli = _rewrite_mcp_tool_verb(command)
        if mcp_cli is not None:
            decision.executed_action = mcp_cli
            decision.compat_status = "rewritten"
            decision.error_tags.append("rewritten_mcp_tool_verb")
            return decision

    cleaned_cmd = command.strip()
    if cleaned_cmd == "systemctl reload telemetry":
        decision.executed_action = "system gateway restart"
        decision.compat_status = "rewritten"
        decision.error_tags.append("rewritten_systemctl_reload")
        return decision
    elif cleaned_cmd == "systemctl status telemetry":
        decision.executed_action = "system gateway status"
        decision.compat_status = "rewritten"
        decision.error_tags.append("rewritten_systemctl_status")
        return decision
    elif cleaned_cmd == "nginx -s reload":
        decision.executed_action = "system gateway restart"
        decision.compat_status = "rewritten"
        decision.error_tags.append("rewritten_nginx_reload")
        return decision

    # Canonicalize file-family write aliases to the single spec verb
    # ``file create``. The AgentToolSpec ``file`` family defines only
    # list/create/read/move/append/delete; models frequently emit
    # ``file write|edit|update|modify --path P --content C`` (all full-content
    # overwrites; ~94 occurrences observed in generated data). Left unrewritten
    # these are silent no-ops at execution (``FileSystemBackend`` has no such
    # subcommand) *and* fail AgentToolSpec validation (degrading the step to an
    # unchecked 'plain'), so a scaffolded or LLM-authored ``file_check``
    # asserting the written value can never pass. Rewriting here -- the single
    # layer shared by validation and execution -- keeps the emitted verb and the
    # executed command in lockstep.
    file_alias_cli = _rewrite_file_write_alias(command)
    if file_alias_cli is not None:
        decision.executed_action = file_alias_cli
        decision.compat_status = "rewritten"
        decision.error_tags.append("rewritten_file_write_alias")
        return decision

    normalized_command, normalized_cron = _normalize_cron_command_quoting(command)
    if normalized_cron:
        decision.error_tags.append("rewritten_cron_command_quoting")
        command = normalized_command

    try:
        tokens = shlex.split(command)
    except ValueError:
        decision.error_tags.append("parse_error")
        return decision

    if not tokens or tokens[0] != "openclaw":
        return decision

    changed = False

    # openclaw message poll --question ... -> --poll-question ...
    if tokens[:3] == ["openclaw", "message", "poll"]:
        if "--question" in tokens and "--poll-question" not in tokens:
            idx = tokens.index("--question")
            tokens[idx] = "--poll-question"
            changed = True
            decision.error_tags.append("rewritten_poll_question")

    # openclaw cron add ... (new CLI may require --name)
    if tokens[:3] == ["openclaw", "cron", "add"]:
        has_schedule = any(flag in tokens for flag in ("--cron", "--schedule", "--every"))
        has_message = any(flag in tokens for flag in ("--message", "--command"))
        if not (has_schedule and has_message) and skip_incompatible:
            decision.compat_status = "skipped_incompatible"
            decision.error_tags.append("incomplete_cron_add")
            decision.skip_reason = (
                "Skipped incompatible openclaw command: "
                "cron add requires a schedule and a message/command."
            )
            return decision
        if "--name" not in tokens:
            job_name = f"job_{hashlib.sha1(command.encode('utf-8')).hexdigest()[:8]}"
            tokens = tokens[:3] + ["--name", job_name] + tokens[3:]
            changed = True
            decision.error_tags.append("rewritten_cron_add_name")

    # openclaw agents add --name X -> openclaw agents add X
    if tokens[:3] == ["openclaw", "agents", "add"] and "--name" in tokens:
        name = _flag(tokens, "--name")
        if name:
            tokens = _remove_flag(tokens, "--name")
            tokens = tokens[:3] + [name] + tokens[3:]
            changed = True
            decision.error_tags.append("rewritten_agents_add_name")

    # openclaw agents set-identity --name X ... -> --agent X ...
    if tokens[:3] == ["openclaw", "agents", "set-identity"]:
        if "--name" in tokens and "--agent" not in tokens:
            name = _flag(tokens, "--name")
            if name:
                tokens = _remove_flag(tokens, "--name")
                tokens = tokens[:3] + ["--agent", name] + tokens[3:]
                changed = True
                decision.error_tags.append("rewritten_agents_set_identity_name")

    # openclaw configure key=value -> openclaw config set key value
    if tokens[:2] == ["openclaw", "configure"] and len(tokens) == 3:
        arg = tokens[2]
        if "=" in arg and not arg.startswith("-"):
            key, value = arg.split("=", 1)
            if key and value:
                tokens = ["openclaw", "config", "set", key, value]
                changed = True
                decision.error_tags.append("rewritten_configure_key_value")

    # Best-effort command alias rewrites from observed failures.
    if tokens[:3] == ["openclaw", "plugins", "remove"]:
        tokens[2] = "uninstall"
        changed = True
        decision.error_tags.append("rewritten_plugins_remove")

    if tokens[:3] == ["openclaw", "channels", "config"]:
        tokens[2] = "set"
        changed = True
        decision.error_tags.append("rewritten_channels_config")

    if tokens[:3] == ["openclaw", "webhooks", "add"]:
        tokens[2] = "create"
        changed = True
        decision.error_tags.append("rewritten_webhooks_add")

    if tokens[:3] == ["openclaw", "security", "set-token"]:
        tokens = tokens[:2] + ["token", "set"] + tokens[3:]
        changed = True
        decision.error_tags.append("rewritten_security_set_token")

    # Explicitly unsupported legacy forms: keep collection running, mark skipped.
    if tokens[:3] == ["openclaw", "message", "react"]:
        if "--target" in tokens and "--message-id" not in tokens and skip_incompatible:
            decision.compat_status = "skipped_incompatible"
            decision.error_tags.append("incompatible_message_react_target_only")
            decision.skip_reason = (
                "Skipped incompatible openclaw command: "
                "message react requires --message-id in current CLI."
            )
            return decision

    if tokens[:3] == ["openclaw", "message", "search"]:
        if "--channel" in tokens and "--guild-id" not in tokens and skip_incompatible:
            decision.compat_status = "skipped_incompatible"
            decision.error_tags.append("incompatible_message_search_missing_guild_id")
            decision.skip_reason = (
                "Skipped incompatible openclaw command: "
                "message search requires --guild-id in current CLI."
            )
            return decision

    # Translate legacy "openclaw" prefixes to new generic families
    if tokens and tokens[0] == "openclaw":
        changed = True
        subcmd = tokens[1] if len(tokens) > 1 else ""
        if subcmd == "message":
            tokens = ["message"] + tokens[2:]
        elif subcmd == "agents":
            tokens = ["agent"] + tokens[2:]
        elif subcmd == "plugins":
            tokens = ["plugin"] + tokens[2:]
        elif subcmd == "channels":
            tokens = ["channel"] + tokens[2:]
        elif subcmd == "skills" and len(tokens) > 2 and tokens[2] == "info":
            tokens = ["plugin", "skill", "info"] + tokens[3:]
        elif subcmd in {"status", "health", "doctor", "gateway", "config", "cron", "security", "devices", "setup"}:
            tokens = ["system"] + tokens[1:]
        else:
            tokens = ["system"] + tokens[1:]

    decision.executed_action = shlex.join(tokens)
    if changed and decision.compat_status == "ok":
        decision.compat_status = "rewritten"
    return decision


def _flag_value(tokens: list[str], *names: str) -> str | None:
    """Return the value following the first of ``names`` present in ``tokens``."""
    for name in names:
        try:
            idx = tokens.index(name)
        except ValueError:
            continue
        if idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def _first_positional(tokens: list[str], skip_values: set[str]) -> str | None:
    """Return the first bare positional token (not a flag or a flag's value)."""
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            # Skip the flag and, if it consumes one, its value.
            if tok in skip_values and i + 1 < len(tokens):
                i += 2
            else:
                i += 1
            continue
        return tok
    return None


def _rewrite_mcp_tool_verb(command: str) -> str | None:
    """Canonicalize an MCP-tool-style verb to the equivalent shell CLI form.

    Mirrors ``backend/mcp_server.py``'s tool->CLI translation so a solution
    trajectory written in the advertised MCP syntax is accepted by AgentToolSpec
    and executed by the environment. Returns the CLI string, or ``None`` when the
    command is not one of the recognized MCP verbs (leave it untouched).

    Handled verbs (flag names accepted leniently):
      * ``run_script --path X [--args "..."]`` -> ``python3 X ...`` / ``bash X ...``
      * ``decompress_archive --path X [--password P] [--output-dir D]``
        -> ``unzip [-P P] [-d D] X`` / ``tar -xzf X [-C D]``
      * ``execute_db_query --db D --query Q`` -> ``sqlite3 D Q``
    """
    stripped = (command or "").strip()
    if not stripped:
        return None
    verb = stripped.split(None, 1)[0]
    if verb not in ("run_script", "decompress_archive", "execute_db_query"):
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if not tokens:
        return None

    if verb == "run_script":
        path = _flag_value(tokens, "--path", "--script-path", "--script_path")
        if path is None:
            path = _first_positional(
                tokens, {"--path", "--script-path", "--script_path", "--args"}
            )
        if not path:
            return None
        args_val = _flag_value(tokens, "--args", "--arg")
        try:
            extra_args = shlex.split(args_val) if args_val else []
        except ValueError:
            extra_args = [args_val] if args_val else []
        interpreter = "bash" if path.endswith(".sh") else "python3"
        return shlex.join([interpreter, path, *extra_args])

    if verb == "decompress_archive":
        path = _flag_value(tokens, "--path", "--archive-path", "--archive_path")
        if path is None:
            path = _first_positional(
                tokens,
                {
                    "--path", "--archive-path", "--archive_path",
                    "--password", "--output-dir", "--output_dir",
                },
            )
        if not path:
            return None
        out_dir = _flag_value(tokens, "--output-dir", "--output_dir")
        if path.endswith((".tar.gz", ".tgz")):
            out = ["tar", "-xzf", path]
            if out_dir:
                out += ["-C", out_dir]
            return shlex.join(out)
        # Default to zip semantics.
        password = _flag_value(tokens, "--password")
        out = ["unzip"]
        if password:
            out += ["-P", password]
        if out_dir:
            out += ["-d", out_dir]
        out.append(path)
        return shlex.join(out)

    if verb == "execute_db_query":
        db = _flag_value(tokens, "--db", "--db-path", "--db_path")
        query = _flag_value(tokens, "--query", "--sql")
        if not db or query is None:
            return None
        return shlex.join(["sqlite3", db, query])

    return None


# File-family write synonyms the models emit that are not AgentToolSpec verbs.
# All share the ``--path P --content C`` full-overwrite shape of ``file create``.
_FILE_WRITE_ALIAS_VERBS = frozenset({"write", "edit", "update", "modify"})


def _rewrite_file_write_alias(command: str) -> str | None:
    """Canonicalize ``file {write,edit,update,modify}`` to ``file create``.

    The AgentToolSpec ``file`` family exposes only
    ``list/create/read/move/append/delete``; ``create`` writes (or overwrites) a
    file's full ``--content``. Models routinely emit the natural-language
    synonyms ``write``/``edit``/``update``/``modify`` with the same
    ``--path``/``--content`` shape. Returns the rewritten CLI string, or ``None``
    when ``command`` is not a file write-alias (leave it untouched).
    """
    stripped = (command or "").strip()
    if not stripped:
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if len(tokens) < 2 or tokens[0] != "file":
        return None
    if tokens[1] not in _FILE_WRITE_ALIAS_VERBS:
        return None
    tokens[1] = "create"
    return shlex.join(tokens)


def classify_error(text: str) -> list[str]:
    """Classify stderr/stdout text into coarse error tags."""
    tags: list[str] = []
    lowered = text.lower()

    if "compatibility config keys detected" in lowered:
        tags.append("compatibility_banner")
    if "invalid config at" in lowered:
        tags.append("invalid_config")
    if "unknown command" in lowered:
        tags.append("unknown_command")
    if "unknown option" in lowered:
        tags.append("unknown_option")
    if "required option" in lowered:
        tags.append("required_option")
    if "unknown channel:" in lowered:
        tags.append("unknown_channel")
    if "unsupported channel:" in lowered:
        tags.append("unsupported_channel")
    if "channel is required (no configured channels detected)" in lowered:
        tags.append("channel_required")
        tags.append("channel_not_configured")
    if "requires at least one configured channel" in lowered:
        tags.append("channel_required")
        tags.append("channel_not_configured")
    if "plugin not found:" in lowered:
        tags.append("plugin_not_found")
    if "command timed out" in lowered:
        tags.append("timeout")
    if "command not found" in lowered:
        tags.append("command_not_found")

    # Preserve insertion order while removing duplicates.
    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def _flag(tokens: list[str], name: str) -> str | None:
    try:
        idx = tokens.index(name)
        return tokens[idx + 1] if idx + 1 < len(tokens) else None
    except ValueError:
        return None


def _remove_flag(tokens: list[str], name: str) -> list[str]:
    try:
        idx = tokens.index(name)
    except ValueError:
        return tokens
    if idx + 1 < len(tokens):
        return tokens[:idx] + tokens[idx + 2 :]
    return tokens[:idx]


def _normalize_cron_command_quoting(command: str) -> tuple[str, bool]:
    """Normalize legacy nested quotes inside `openclaw cron add --command '...'`.

    Some generated GT commands embed a quoted message inside a single-quoted
    `--command` argument, which breaks shell parsing in real CLI mode.
    """
    if not command.strip().startswith("openclaw cron add "):
        return command, False

    match = re.search(r"--command\s+'(.*)'\s*$", command)
    if not match:
        return command, False

    prefix = command[: match.start()].rstrip()
    inner = match.group(1)
    rebuilt = f"{prefix} --command {shlex.quote(inner)}"
    return rebuilt, rebuilt != command
