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

"""Message skill implementation with local in-memory state and real platform integration."""

from __future__ import annotations

import json
import os
import re
import shlex
import urllib.parse
import urllib.request
import uuid
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return default


def _get_multi_arg(args: list[str], flag: str) -> list[str]:
    results = []
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            results.append(args[i + 1])
    return results


def _split_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class MessageSkill(Skill):
    """Message skill handles sending, broadcasting, polling, and searching messages."""

    SPEC = SkillSpec(
        name="message",
        category="comms",
        summary="Multi-agent messaging substrate (send/broadcast/poll/search/react).",
        prefixes=("message",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "message send",
                "Send a message to a target on a channel.",
                args={
                    "--channel": {"type": "string"},
                    "--target": {"type": "string"},
                    "--message": {"type": "string"},
                    "--media": {"type": "string"},
                },
                required=("--channel", "--target"),
                emits_effects=("messages",),
            ),
            CommandSpec(
                "message broadcast",
                "Broadcast a message to a channel.",
                args={
                    "--channel": {"type": "string"},
                    "--message": {"type": "string"},
                },
                required=("--channel", "--message"),
                emits_effects=("messages",),
            ),
            CommandSpec(
                "message search",
                "Search prior messages by query.",
                args={"--query": {"type": "string"}},
            ),
            CommandSpec(
                "message poll",
                "Create a poll with a question and options.",
                args={
                    "--channel": {"type": "string"},
                    "--question": {"type": "string"},
                    "--options": {"type": "string"},
                },
                required=("--question",),
                emits_effects=("messages",),
            ),
            CommandSpec(
                "message react",
                "React to a message by id with an emoji.",
                args={"--id": {"type": "string"}, "--emoji": {"type": "string"}},
                required=("--id", "--emoji"),
            ),
        ),
        smoke=(
            "message send --channel slack --target alice --message hi",
        ),
    )

    _snapshot_attrs = ("_messages",)

    def __init__(self) -> None:
        super().__init__(prefixes=("message",))
        self._messages: list[dict[str, Any]] = []
        self._state_dir: str = ""
        self._initialized = False
        self._enable_online_data = False
        self._slack_token = ""
        self._slack_webhook_url = ""
        self._telegram_token = ""
        self._telegram_chat_id = ""

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._state_dir = state_dir
        self._messages = []
        self._initialized = True
        
        # Look for online configs
        val = str(env_vars.get("CLASHARNESS_ENABLE_ONLINE_DATA", "")).lower()
        self._enable_online_data = val in {"1", "true", "yes", "on"}
        
        self._slack_token = env_vars.get("SLACK_BOT_TOKEN", "").strip()
        self._slack_webhook_url = env_vars.get("SLACK_WEBHOOK_URL", "").strip()
        self._telegram_token = env_vars.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._telegram_chat_id = env_vars.get("TELEGRAM_CHAT_ID", "").strip()

    def execute(self, command: str) -> CommandResult:
        parts = shlex.split(command.strip())
        if not parts or parts[0] != "message":
            return CommandResult(stdout="", stderr="Not a message command", exit_code=1)

        if len(parts) < 2:
            return CommandResult(
                stdout="Usage: message <send|broadcast|poll|react|search|read|delete>",
                stderr="",
                exit_code=0,
            )

        action = parts[1]
        args = parts[2:]

        handlers = {
            "send": self._action_send,
            "broadcast": self._action_broadcast,
            "search": self._action_search,
            "poll": self._action_poll,
            "react": self._action_react,
        }

        handler = handlers.get(action)
        if handler is None:
            return CommandResult(
                stdout="", stderr=f"Unknown message action: {action}", exit_code=1
            )
        return handler(args)

    def _action_send(self, args: list[str]) -> CommandResult:
        target = _get_arg(args, "--target", "")
        message = _get_arg(args, "--message", "")
        channel = _get_arg(args, "--channel", "")
        media = _get_arg(args, "--media")

        if not channel:
            return CommandResult(stdout="", stderr="--channel is required", exit_code=1)
        if not target:
            return CommandResult(stdout="", stderr="--target is required", exit_code=1)
        if not message and not media:
            return CommandResult(stdout="", stderr="--message or --media is required", exit_code=1)

        msg_record = {
            "id": uuid.uuid4().hex[:8],
            "target": target,
            "message": message,
            "channel": channel,
            "media": media,
            "type": "media" if media else "text",
        }
        self._messages.append(msg_record)

        # Real mode online execution if enabled
        online_success = False
        online_err = ""
        if self._enable_online_data:
            online_success, online_err = self._send_online_message(channel, target, message)

        stdout = f"Message sent to {target}" + (f" via {channel}" if channel else "") + "."
        if self._enable_online_data:
            if online_success:
                stdout += " (Delivered to real chat platform)"
            else:
                stdout += f" (Online delivery failed: {online_err})"

        return CommandResult(
            stdout=stdout,
            stderr="",
            exit_code=0,
            state_changes={"messages": [msg_record]},
        )

    def _action_broadcast(self, args: list[str]) -> CommandResult:
        message = _get_arg(args, "--message", "") or ""
        channel = _get_arg(args, "--channel", "") or ""
        target_list: list[str] = []
        if "--targets" in args:
            idx = args.index("--targets") + 1
            while idx < len(args) and not args[idx].startswith("--"):
                target_list.append(args[idx])
                idx += 1
        if len(target_list) == 1 and "," in target_list[0]:
            target_list = _split_csv(target_list[0])
        # Channel-routed form. This is the documented syntax
        # (``message broadcast --channel NAME --message TEXT``) and the ONLY form
        # the agent-facing spec exposes -- ``agent_tools.v1.json`` declares just
        # ``channel``/``message``, so ``--targets`` is unreachable from the agent
        # path. Without this branch every agent broadcast fell through the
        # ``--targets`` loop with an empty list and returned exit 0 having
        # recorded nothing: a silent no-op that never reached ``messages_sent``.
        # Uses the same ``target or channel`` convention as
        # ``Environment._record_subagent_message``.
        if not target_list and channel:
            target_list = [channel]
        # Fail loudly instead of returning a successful zero-target broadcast.
        if not target_list:
            return CommandResult(
                stdout="", stderr="--channel or --targets is required", exit_code=1
            )
        if not message:
            return CommandResult(stdout="", stderr="--message is required", exit_code=1)

        sent_records: list[dict[str, Any]] = []
        for t in target_list:
            msg_record = {
                "id": uuid.uuid4().hex[:8],
                "target": t,
                "message": message,
                # Carry the channel like ``send`` does, so a broadcast is also
                # findable via ``message search --channel`` (which filters on
                # this field and therefore never matched a broadcast before).
                "channel": channel,
                "type": "text",
            }
            self._messages.append(msg_record)
            sent_records.append(msg_record)
            if self._enable_online_data:
                # Previously passed the literal "broadcast" as the channel, which
                # resolves to no real channel; the actual channel is now known.
                self._send_online_message(channel or "broadcast", t, message)

        # Feed the single messages_sent sink uniformly (same as `send`) so a
        # broadcast is visible to evaluation, not silently dropped.
        return CommandResult(
            stdout=f"Broadcast sent to {len(target_list)} targets.",
            stderr="",
            exit_code=0,
            state_changes={"messages": sent_records},
        )

    def _action_search(self, args: list[str]) -> CommandResult:
        query = _get_arg(args, "--query", "")
        channel = _get_arg(args, "--channel", "")
        results = [
            m for m in self._messages
            if query.lower() in m.get("message", "").lower()
            and (not channel or m.get("channel", "") == channel)
        ]
        if not results:
            return CommandResult(
                stdout=f"No messages found matching '{query}'.", stderr="", exit_code=0
            )
        lines = [f"Found {len(results)} message(s):"]
        for m in results:
            lines.append(f"  [{m['id']}] to {m['target']}: {m['message']}")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _action_poll(self, args: list[str]) -> CommandResult:
        question = _get_arg(args, "--poll-question") or _get_arg(args, "--question", "")
        poll_options = _get_multi_arg(args, "--poll-option")
        options = poll_options if poll_options else _split_csv(_get_arg(args, "--options", "") or "")
        target = _get_arg(args, "--target", "") or ""
        # Channel-routed form. ``message poll --channel NAME --question TEXT
        # --options A,B`` is the documented syntax and the only one the
        # agent-facing spec exposes, but ``--channel`` used to be dropped on the
        # floor: the record was stamped with an empty target and carried no
        # channel at all, so a doc-following poll was invisible to every
        # target/channel-scoped check. Same ``target or channel`` convention as
        # ``send``/``broadcast`` and ``Environment._record_subagent_message``.
        channel = _get_arg(args, "--channel", "") or ""
        target = target or channel
        msg_record = {
            "id": uuid.uuid4().hex[:8],
            "target": target,
            "channel": channel,
            "type": "poll",
            "question": question,
            "options": options,
        }
        self._messages.append(msg_record)

        if self._enable_online_data:
            poll_text = f"POLL: {question}\nOptions:\n" + "\n".join(f"- {o}" for o in options)
            # Previously passed the literal "poll" as the channel, which resolves
            # to no real channel; the actual channel is now known.
            self._send_online_message(channel or "poll", target, poll_text)

        # Feed the single messages_sent sink uniformly (same as `send`).
        return CommandResult(
            stdout=f"Poll created: {question}",
            stderr="",
            exit_code=0,
            state_changes={"messages": [msg_record]},
        )

    def _action_react(self, args: list[str]) -> CommandResult:
        target = _get_arg(args, "--target", "") or ""
        emoji = _get_arg(args, "--emoji", "👍")
        channel = _get_arg(args, "--channel", "") or ""
        # ``message react --channel NAME --message-id ID --emoji EMOJI`` is the
        # documented syntax, but ``--message-id`` used to be dropped entirely:
        # the reaction could not be tied back to the message it reacted to, and
        # a doc-following call (which passes no ``--target``) produced a record
        # with an empty target. ``--target`` stays supported as the historical
        # spelling; the destination falls back to the channel using the same
        # convention as ``send``/``broadcast``/``poll``.
        message_id = _get_arg(args, "--message-id", "") or ""
        target = target or channel

        # Real platform reaction could be implemented here if needed.

        # A reaction is a communication effect: feed the single messages_sent
        # sink uniformly (same as send/broadcast/poll) so it is visible to
        # evaluation. It is intentionally NOT added to the searchable message
        # pool (a reaction is metadata on an existing message, not a new one);
        # the empty "message" keeps it out of content/semantic transcripts.
        react_record = {
            "id": uuid.uuid4().hex[:8],
            "target": target,
            "emoji": emoji,
            "channel": channel,
            "message_id": message_id,
            "type": "reaction",
            "message": "",
        }
        return CommandResult(
            stdout=f"Reacted with {emoji} to message "
            + (f"{message_id}" if message_id else f"from {target}")
            + (f" on {channel}" if channel else "") + ".",
            stderr="",
            exit_code=0,
            state_changes={"messages": [react_record]},
        )

    def _send_online_message(self, channel: str, target: str, message: str) -> tuple[bool, str]:
        """Send message directly to Telegram or Slack endpoints in real mode."""
        # Slack Webhook URL mode
        if self._slack_webhook_url:
            try:
                payload = json.dumps({"text": f"[{target} / {channel}] {message}"}).encode("utf-8")
                req = urllib.request.Request(
                    self._slack_webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status in (200, 204):
                        return True, ""
                    return False, f"HTTP Status {response.status}"
            except Exception as e:
                return False, str(e)

        # Telegram Bot API mode
        if self._telegram_token and self._telegram_chat_id:
            try:
                payload = json.dumps({
                    "chat_id": self._telegram_chat_id,
                    "text": f"[{target} / {channel}] {message}"
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{self._telegram_token}/sendMessage",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    if res_data.get("ok"):
                        return True, ""
                    return False, res_data.get("description", "Unknown Telegram error")
            except Exception as e:
                return False, str(e)

        # No configuration provided
        return False, "No active channel credentials configured (SLACK_WEBHOOK_URL or TELEGRAM_BOT_TOKEN)"

    def add_mock_message(self, channel: str, target: str, message: str, media: str | None = None) -> None:
        msg_record = {
            "id": uuid.uuid4().hex[:8],
            "target": target,
            "message": message,
            "channel": channel,
            "media": media,
            "type": "media" if media else "text",
        }
        self._messages.append(msg_record)

    def get_state(self) -> dict[str, Any]:

        return {"messages": self._messages}

    def cleanup(self) -> None:
        self._messages = []
