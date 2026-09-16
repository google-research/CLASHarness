#!/usr/bin/env python3

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

"""Discover required channels from generated tasks and login in one pass.

Examples:
  # Show channels used by dev split
  python scripts/login_required_channels.py --split dev --print-only

  # Login all channels required by dev split (interactive confirmation)
  python scripts/login_required_channels.py --split dev

  # Non-interactive run with extra args per channel
  python scripts/login_required_channels.py --split all --yes \
      --extra-args-file channel_login_args.json

  # Include Google Calendar OAuth bootstrap for calendar tasks
  python scripts/login_required_channels.py --split dev \
      --google-client-secret-file ~/.openclaw/client_secret.json

  # Include Google Calendar + Email + Tasks online bootstrap checks
  python scripts/login_required_channels.py --split dev \
      --google-client-secret-file ~/.openclaw/client_secret.json \
      --google-email-client-secret-file ~/.openclaw/client_secret.json \
      --google-tasks-token-file ~/.openclaw/google_tasks_token.json \
      --email-provider auto

`channel_login_args.json` format:
{
  "telegram": ["--bot-token", "xxx"],
  "discord": "--token abc --guild-id 123"
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_KNOWN_CHANNELS = {
    "telegram",
    "whatsapp",
    "discord",
    "irc",
    "googlechat",
    "slack",
    "signal",
    "imessage",
    "feishu",
    "nostr",
    "msteams",
    "mattermost",
    "nextcloud-talk",
    "matrix",
    "bluebubbles",
    "line",
    "zalo",
    "zalouser",
    "synology-chat",
    "tlon",
}

_CHANNEL_ALIASES = {
    "google-chat": "googlechat",
    "google_chat": "googlechat",
    "google chat": "googlechat",
    "nextcloud_talk": "nextcloud-talk",
    "nextcloud talk": "nextcloud-talk",
    "ms-teams": "msteams",
    "microsoft-teams": "msteams",
    "microsoft teams": "msteams",
    "synology_chat": "synology-chat",
    "synology chat": "synology-chat",
}


def _normalize_channel(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        return ""
    normalized = _CHANNEL_ALIASES.get(value, value)
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract required channels from tasks and run "
            "`openclaw channels login --channel <name>` in batch."
        )
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="dev",
        help="Dataset split to scan (default: dev).",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Specific task_id(s) to scan. Can be repeated.",
    )
    parser.add_argument(
        "--channels",
        default="",
        help="Comma-separated channels to force-include.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print detected channels, do not run login commands.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run all login commands without per-channel confirmation.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch when a channel login command fails.",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Optional OpenClaw profile name (passed as --profile).",
    )
    parser.add_argument(
        "--dev-profile",
        action="store_true",
        help="Pass global --dev to OpenClaw.",
    )
    parser.add_argument(
        "--extra-args-file",
        default="",
        help=(
            "Optional JSON file mapping channel -> extra CLI args list/string. "
            "These args are appended to each login command."
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        default="clasharness/data/datasets",
        help="Path to datasets directory (default: clasharness/data/datasets).",
    )
    parser.add_argument(
        "--tasks-dir",
        default="clasharness/data/tasks",
        help="Path to task specs directory (default: clasharness/data/tasks).",
    )
    parser.add_argument(
        "--skip-calendar-login",
        action="store_true",
        help="Do not run Google Calendar OAuth bootstrap even if calendar tasks are detected.",
    )
    parser.add_argument(
        "--skip-email-login",
        action="store_true",
        help="Do not run email provider bootstrap even if email tasks are detected.",
    )
    parser.add_argument(
        "--skip-tasks-login",
        action="store_true",
        help="Do not run Google Tasks OAuth bootstrap even if tasks tasks are detected.",
    )
    parser.add_argument(
        "--google-client-secret-file",
        default="",
        help=(
            "Path to Google OAuth client secret JSON. "
            "If omitted, fallback to CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE."
        ),
    )
    parser.add_argument(
        "--google-token-file",
        default="",
        help=(
            "Path for Google OAuth token JSON. "
            "If omitted, fallback to CLASHARNESS_GOOGLE_TOKEN_FILE or "
            "~/.openclaw/google_calendar_token.json."
        ),
    )
    parser.add_argument(
        "--google-calendar-id",
        default="",
        help="Calendar ID for bootstrap check (default: env or primary).",
    )
    parser.add_argument(
        "--google-timezone",
        default="",
        help="Timezone for calendar provider (default: env or UTC).",
    )
    parser.add_argument(
        "--google-scopes",
        default="",
        help=(
            "Comma-separated Google OAuth scopes for bootstrap "
            "(default: env or https://www.googleapis.com/auth/calendar)."
        ),
    )
    parser.add_argument(
        "--google-email-client-secret-file",
        default="",
        help=(
            "Path to Google OAuth client secret JSON for Gmail bootstrap. "
            "If omitted, fallback to --google-client-secret-file then "
            "CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE."
        ),
    )
    parser.add_argument(
        "--email-provider",
        choices=["auto", "google_api", "himalaya", "mock"],
        default="auto",
        help="Email online provider bootstrap mode (default: auto).",
    )
    parser.add_argument(
        "--google-tasks-token-file",
        default="",
        help=(
            "Path for Google Tasks OAuth token JSON. "
            "If omitted, fallback to CLASHARNESS_GOOGLE_TASKS_TOKEN_FILE or "
            "~/.openclaw/google_tasks_token.json."
        ),
    )
    return parser.parse_args()


def _read_task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_id:
        return list(dict.fromkeys(t.strip() for t in args.task_id if t.strip()))

    datasets_dir = Path(args.datasets_dir)
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    task_ids: list[str] = []
    for split in splits:
        split_path = datasets_dir / f"{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        task_ids.extend(
            [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        )
    return list(dict.fromkeys(task_ids))


def _extract_channel_from_command(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if len(tokens) < 2 or tokens[0] != "openclaw":
        return None

    for i, tok in enumerate(tokens):
        if tok in {"--channel", "--reply-channel"} and i + 1 < len(tokens):
            return _normalize_channel(tokens[i + 1])

    # Fallback for potential positional login syntax.
    if tokens[:3] == ["openclaw", "channels", "login"] and len(tokens) >= 4:
        candidate = tokens[3]
        if not candidate.startswith("-"):
            return _normalize_channel(candidate)
    return None


def _extract_channels_from_obj(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).lower()
            if key in {"channel", "reply_channel", "target_channel"} and isinstance(v, str):
                ch = _normalize_channel(v)
                if ch:
                    found.add(ch)
            elif key == "channels":
                if isinstance(v, str):
                    ch = _normalize_channel(v)
                    if ch:
                        found.add(ch)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            ch = _normalize_channel(item)
                            if ch:
                                found.add(ch)
            found.update(_extract_channels_from_obj(v))
    elif isinstance(value, list):
        for item in value:
            found.update(_extract_channels_from_obj(item))
    return found


def _extract_channels_from_instruction(instruction: str) -> set[str]:
    found: set[str] = set()
    text = instruction.lower()
    if not text:
        return found
    for channel in _KNOWN_CHANNELS:
        pattern = (
            r"\b(?:on|via|through|using|to|in)\s+"
            + re.escape(channel)
            + r"\b|\bchannel\s+"
            + re.escape(channel)
            + r"\b"
        )
        if re.search(pattern, text):
            found.add(channel)
    return found


def _collect_channels(
    task_ids: list[str],
    tasks_dir: Path,
) -> tuple[list[str], Counter, int, int, int]:
    counts: Counter[str] = Counter()
    calendar_task_count = 0
    email_task_count = 0
    tasks_task_count = 0
    for task_id in task_ids:
        specs = tasks_dir / task_id / "specs.json"
        if not specs.exists():
            continue
        try:
            obj = json.loads(specs.read_text())
        except Exception:
            continue
        cmds = obj.get("ground_truth", {}).get("solution_commands", [])
        if not isinstance(cmds, list):
            continue
        task_uses_calendar = False
        task_uses_email = False
        task_uses_tasks = False
        domains = obj.get("domains", [])
        if isinstance(domains, list) and any(str(d).strip().lower() == "calendar" for d in domains):
            task_uses_calendar = True
        if isinstance(domains, list) and any(str(d).strip().lower() == "email" for d in domains):
            task_uses_email = True
        if isinstance(domains, list) and any(str(d).strip().lower() == "tasks" for d in domains):
            task_uses_tasks = True
        seen_for_task: set[str] = set()
        for cmd in cmds:
            if not isinstance(cmd, str):
                continue
            channel = _extract_channel_from_command(cmd)
            if channel:
                seen_for_task.add(channel)
            stripped = cmd.strip()
            if stripped.startswith("calendar ") or stripped.startswith("gcalcli "):
                task_uses_calendar = True
            if stripped.startswith("email "):
                task_uses_email = True
            if stripped.startswith("tasks "):
                task_uses_tasks = True

        # Additional sources: task data and instruction text.
        task_data = obj.get("data", {})
        seen_for_task.update(_extract_channels_from_obj(task_data))
        instruction = obj.get("instruction", "")
        if isinstance(instruction, str):
            seen_for_task.update(_extract_channels_from_instruction(instruction))

        if task_uses_calendar:
            calendar_task_count += 1
        if task_uses_email:
            email_task_count += 1
        if task_uses_tasks:
            tasks_task_count += 1

        for ch in seen_for_task:
            counts[ch] += 1
    channels = sorted(counts.keys())
    return channels, counts, calendar_task_count, email_task_count, tasks_task_count


def _load_extra_args(path: str) -> dict[str, list[str]]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("--extra-args-file must contain a JSON object")
    result: dict[str, list[str]] = {}
    for channel, value in raw.items():
        key = _normalize_channel(str(channel))
        if isinstance(value, str):
            result[key] = shlex.split(value)
        elif isinstance(value, list):
            result[key] = [str(v) for v in value]
        else:
            raise ValueError(
                f"Invalid extra args for channel '{channel}': must be string or list"
            )
    return result


def _build_openclaw_prefix(args: argparse.Namespace) -> list[str]:
    cmd = ["openclaw"]
    if args.dev_profile:
        cmd.append("--dev")
    if args.profile:
        cmd.extend(["--profile", args.profile])
    return cmd


def _run_login_batch(
    channels: list[str],
    args: argparse.Namespace,
    extra_args: dict[str, list[str]],
) -> int:
    prefix = _build_openclaw_prefix(args)
    failed = 0

    for i, channel in enumerate(channels, 1):
        cmd = prefix + ["channels", "login", "--channel", channel] + extra_args.get(
            channel, []
        )
        printable = shlex.join(cmd)

        if not args.yes:
            answer = input(f"[{i}/{len(channels)}] Run {printable}? [Y/n] ").strip().lower()
            if answer in {"n", "no"}:
                print(f"  - skipped {channel}")
                continue

        print(f"\n>>> {printable}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failed += 1
            print(f"  - FAILED ({proc.returncode}) for channel '{channel}'")
            if args.stop_on_error:
                return failed
        else:
            print(f"  - OK: {channel}")

    return failed


def _build_calendar_oauth_env(
    args: argparse.Namespace,
) -> tuple[dict[str, str] | None, str | None]:
    env = dict(os.environ)
    client_secret = (
        args.google_client_secret_file.strip()
        or env.get("CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE", "").strip()
    )
    if not client_secret:
        return None, (
            "missing Google OAuth client secret. "
            "Provide --google-client-secret-file or set "
            "CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE."
        )
    client_secret = os.path.expanduser(client_secret)
    if not os.path.isfile(client_secret):
        return None, f"client secret file not found: {client_secret}"

    token_file = (
        args.google_token_file.strip()
        or env.get("CLASHARNESS_GOOGLE_TOKEN_FILE", "").strip()
        or os.path.expanduser("~/.openclaw/google_calendar_token.json")
    )
    calendar_id = (
        args.google_calendar_id.strip()
        or env.get("CLASHARNESS_GOOGLE_CALENDAR_ID", "").strip()
        or "primary"
    )
    timezone = (
        args.google_timezone.strip()
        or env.get("CLASHARNESS_GOOGLE_TIMEZONE", "").strip()
        or "UTC"
    )
    scopes = (
        args.google_scopes.strip()
        or env.get("CLASHARNESS_GOOGLE_SCOPES", "").strip()
        or "https://www.googleapis.com/auth/calendar"
    )

    env["CLASHARNESS_ENABLE_ONLINE_DATA"] = "1"
    env["CLASHARNESS_STRICT_ONLINE_DATA"] = "1"
    env["CLASHARNESS_CALENDAR_PROVIDER"] = "google_api"
    env["CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE"] = client_secret
    env["CLASHARNESS_GOOGLE_TOKEN_FILE"] = token_file
    env["CLASHARNESS_GOOGLE_CALENDAR_ID"] = calendar_id
    env["CLASHARNESS_GOOGLE_TIMEZONE"] = timezone
    env["CLASHARNESS_GOOGLE_SCOPES"] = scopes
    return env, None


def _run_calendar_oauth_bootstrap(args: argparse.Namespace, prompt_index: int | None = None) -> int:
    env, env_error = _build_calendar_oauth_env(args)
    if env is None:
        print(
            "\n[calendar] skipped: "
            + (env_error or "missing Google OAuth client secret.")
        )
        return 1

    py_code = (
        "import os\n"
        "from clasharness.skills.impl.calendar_skill import (\n"
        "    _build_google_settings,\n"
        "    _build_google_calendar_service,\n"
        ")\n"
        "settings=_build_google_settings(os.environ)\n"
        "service=_build_google_calendar_service(settings)\n"
        "resp=service.calendarList().list(maxResults=1).execute()\n"
        "items=resp.get('items', []) if isinstance(resp, dict) else []\n"
        "print(\n"
        "    f\"Google Calendar OAuth OK: calendars_visible={len(items)} \"\n"
        "    f\"token_file={settings.get('token_file')}\"\n"
        ")\n"
    )
    cmd = [sys.executable, "-c", py_code]
    printable = shlex.join(cmd)

    if not args.yes:
        prefix = f"[{prompt_index}] " if prompt_index is not None else ""
        answer = input(
            f"{prefix}Run Google Calendar OAuth bootstrap ({printable})? [Y/n] "
        ).strip().lower()
        if answer in {"n", "no"}:
            print("  - skipped calendar OAuth bootstrap")
            return 0

    print(f"\n>>> {printable}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"  - FAILED ({proc.returncode}) for Google Calendar OAuth bootstrap")
        return 1
    print("  - OK: google calendar oauth")
    return 0


def _build_email_google_oauth_env(
    args: argparse.Namespace,
) -> tuple[dict[str, str] | None, str | None]:
    env = dict(os.environ)
    client_secret = (
        args.google_email_client_secret_file.strip()
        or args.google_client_secret_file.strip()
        or env.get("CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE", "").strip()
    )
    if not client_secret:
        return None, (
            "missing Google OAuth client secret for Gmail. "
            "Provide --google-email-client-secret-file (or --google-client-secret-file)."
        )
    client_secret = os.path.expanduser(client_secret)
    if not os.path.isfile(client_secret):
        return None, f"client secret file not found: {client_secret}"

    token_file = (
        env.get("CLASHARNESS_GOOGLE_TOKEN_FILE", "").strip()
        or os.path.expanduser("~/.openclaw/google_email_token.json")
    )
    scopes = (
        env.get("CLASHARNESS_GOOGLE_GMAIL_SCOPES", "").strip()
        or "https://www.googleapis.com/auth/gmail.modify,"
        "https://www.googleapis.com/auth/gmail.send"
    )
    env["CLASHARNESS_ENABLE_ONLINE_DATA"] = "1"
    env["CLASHARNESS_STRICT_ONLINE_DATA"] = "1"
    env["CLASHARNESS_EMAIL_PROVIDER"] = "google_api"
    env["CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE"] = client_secret
    env["CLASHARNESS_GOOGLE_TOKEN_FILE"] = token_file
    env["CLASHARNESS_GOOGLE_GMAIL_SCOPES"] = scopes
    return env, None


def _run_google_email_bootstrap(args: argparse.Namespace) -> int:
    env, env_error = _build_email_google_oauth_env(args)
    if env is None:
        print("\n[email/google_api] unavailable: " + (env_error or "missing config"))
        return 1

    py_code = (
        "import os\n"
        "from clasharness.skills.impl.email_skill import (\n"
        "    _build_google_gmail_settings,\n"
        "    _build_google_gmail_service,\n"
        ")\n"
        "settings=_build_google_gmail_settings(os.environ)\n"
        "service=_build_google_gmail_service(settings, allow_interactive=True)\n"
        "resp=service.users().messages().list(userId=settings.get('user_id','me'), maxResults=1).execute()\n"
        "msgs=resp.get('messages', []) if isinstance(resp, dict) else []\n"
        "print(\n"
        "    f\"Google Gmail OAuth OK: messages_visible={len(msgs)} \"\n"
        "    f\"token_file={settings.get('token_file')}\"\n"
        ")\n"
    )
    cmd = [sys.executable, "-c", py_code]
    printable = shlex.join(cmd)
    print(f"\n>>> {printable}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"  - FAILED ({proc.returncode}) for Google Gmail OAuth bootstrap")
        return 1
    print("  - OK: google gmail oauth")
    return 0


def _check_himalaya_provider() -> tuple[bool, str]:
    bin_path = shutil.which("himalaya")
    if not bin_path:
        return False, "himalaya binary not found in PATH"
    env = dict(os.environ)
    account = env.get("CLASHARNESS_HIMALAYA_ACCOUNT", "").strip()
    config_file = (
        env.get("CLASHARNESS_HIMALAYA_CONFIG_FILE", "").strip()
        or os.path.expanduser("~/.config/himalaya/config.toml")
    )
    has_config = bool(account) or os.path.isfile(os.path.expanduser(config_file))
    if not has_config:
        return False, (
            "himalaya found but account/config missing "
            "(set CLASHARNESS_HIMALAYA_ACCOUNT or ~/.config/himalaya/config.toml)"
        )
    return True, f"himalaya ready (bin={bin_path}, account={account or '(from config)'})"


def _run_email_provider_bootstrap(args: argparse.Namespace, prompt_index: int | None = None) -> int:
    provider = args.email_provider
    prefix = f"[{prompt_index}] " if prompt_index is not None else ""
    if not args.yes:
        answer = input(
            f"{prefix}Run email provider bootstrap (provider={provider})? [Y/n] "
        ).strip().lower()
        if answer in {"n", "no"}:
            print("  - skipped email bootstrap")
            return 0

    if provider == "mock":
        print("\n[email] provider=mock, skipped online bootstrap")
        return 0

    if provider in {"google_api", "auto"}:
        rc = _run_google_email_bootstrap(args)
        if rc == 0:
            return 0
        if provider == "google_api":
            return rc
        print("[email/auto] google_api unavailable, checking himalaya...")

    if provider in {"himalaya", "auto"}:
        ok, detail = _check_himalaya_provider()
        if ok:
            print(f"[email/himalaya] {detail}")
            return 0
        print(f"[email/himalaya] unavailable: {detail}")
        return 1

    return 0


def _build_tasks_google_oauth_env(
    args: argparse.Namespace,
) -> tuple[dict[str, str] | None, str | None]:
    env = dict(os.environ)
    client_secret = (
        args.google_client_secret_file.strip()
        or env.get("CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE", "").strip()
    )
    if not client_secret:
        return None, (
            "missing Google OAuth client secret for Tasks. "
            "Provide --google-client-secret-file."
        )
    client_secret = os.path.expanduser(client_secret)
    if not os.path.isfile(client_secret):
        return None, f"client secret file not found: {client_secret}"

    token_file = (
        args.google_tasks_token_file.strip()
        or env.get("CLASHARNESS_GOOGLE_TASKS_TOKEN_FILE", "").strip()
        or os.path.expanduser("~/.openclaw/google_tasks_token.json")
    )
    scopes = (
        env.get("CLASHARNESS_GOOGLE_TASKS_SCOPES", "").strip()
        or "https://www.googleapis.com/auth/tasks"
    )
    env["CLASHARNESS_ENABLE_ONLINE_DATA"] = "1"
    env["CLASHARNESS_STRICT_ONLINE_DATA"] = "1"
    env["CLASHARNESS_TASKS_PROVIDER"] = "google_api"
    env["CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE"] = client_secret
    env["CLASHARNESS_GOOGLE_TASKS_TOKEN_FILE"] = token_file
    env["CLASHARNESS_GOOGLE_TASKS_SCOPES"] = scopes
    return env, None


def _run_google_tasks_bootstrap(args: argparse.Namespace) -> int:
    env, env_error = _build_tasks_google_oauth_env(args)
    if env is None:
        print("\n[tasks/google_api] unavailable: " + (env_error or "missing config"))
        return 1

    py_code = (
        "import os\n"
        "from clasharness.skills.impl.tasks_skill import (\n"
        "    _build_google_tasks_settings,\n"
        "    _build_google_tasks_service,\n"
        ")\n"
        "settings=_build_google_tasks_settings(os.environ)\n"
        "service=_build_google_tasks_service(settings, allow_interactive=True)\n"
        "resp=service.tasklists().list(maxResults=1).execute()\n"
        "items=resp.get('items', []) if isinstance(resp, dict) else []\n"
        "print(\n"
        "    f\"Google Tasks OAuth OK: tasklists_visible={len(items)} \"\n"
        "    f\"token_file={settings.get('token_file')}\"\n"
        ")\n"
    )
    cmd = [sys.executable, "-c", py_code]
    printable = shlex.join(cmd)
    print(f"\n>>> {printable}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"  - FAILED ({proc.returncode}) for Google Tasks OAuth bootstrap")
        return 1
    print("  - OK: google tasks oauth")
    return 0


def main() -> int:
    args = _parse_args()
    task_ids = _read_task_ids(args)
    tasks_dir = Path(args.tasks_dir)
    channels, counts, calendar_task_count, email_task_count, tasks_task_count = _collect_channels(task_ids, tasks_dir)

    manual = [_normalize_channel(x) for x in args.channels.split(",") if x.strip()]
    all_channels = sorted(set(channels).union(manual))

    print(f"Scanned tasks: {len(task_ids)}")
    print(f"Detected channels: {', '.join(all_channels) if all_channels else '(none)'}")
    print(f"Detected calendar tasks: {calendar_task_count}")
    print(f"Detected email tasks: {email_task_count}")
    print(f"Detected tasks tasks: {tasks_task_count}")
    if counts:
        print("Channel coverage (tasks):")
        for ch in all_channels:
            print(f"  - {ch:<10} {counts.get(ch, 0)}")

    should_try_calendar_login = calendar_task_count > 0 and not args.skip_calendar_login
    should_try_email_login = email_task_count > 0 and not args.skip_email_login
    should_try_tasks_login = tasks_task_count > 0 and not args.skip_tasks_login

    if args.print_only or (
        not all_channels
        and not should_try_calendar_login
        and not should_try_email_login
        and not should_try_tasks_login
    ):
        return 0

    extra_args = _load_extra_args(args.extra_args_file)
    failed = _run_login_batch(all_channels, args, extra_args) if all_channels else 0
    if should_try_calendar_login:
        prompt_index = len(all_channels) + 1 if all_channels else 1
        failed += _run_calendar_oauth_bootstrap(args, prompt_index=prompt_index)
    if should_try_email_login:
        prompt_index = len(all_channels) + 1
        if should_try_calendar_login:
            prompt_index += 1
        failed += _run_email_provider_bootstrap(args, prompt_index=prompt_index)
    if should_try_tasks_login:
        prompt_index = len(all_channels) + 1
        if should_try_calendar_login:
            prompt_index += 1
        if should_try_email_login:
            prompt_index += 1
        if not args.yes:
            answer = input(
                f"[{prompt_index}] Run Google Tasks OAuth bootstrap? [Y/n] "
            ).strip().lower()
            if answer in {"n", "no"}:
                print("  - skipped tasks OAuth bootstrap")
            else:
                failed += _run_google_tasks_bootstrap(args)
        else:
            failed += _run_google_tasks_bootstrap(args)

    print("\nDone.")
    print(
        "Recommended checks:\n"
        "  openclaw status --json\n"
        "  openclaw channels list --json\n"
        "  test -f ~/.openclaw/google_calendar_token.json && echo 'google calendar token OK'\n"
        "  test -f ~/.openclaw/google_email_token.json && echo 'google email token OK'\n"
        "  test -f ~/.openclaw/google_tasks_token.json && echo 'google tasks token OK'"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
