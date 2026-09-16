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

"""Rewrite task instructions to sound more like real user input.

This script is intentionally post-generation:
- Keep solution actions deterministic and controllable.
- Rewrite only the question/instruction text via GPT (or mock mode).
- Apply changes in each task's own specs.json while keeping original structure.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class RewriteResult:
    task_id: str
    spec_path: str
    old_question: str
    new_question: str
    status: str
    reason: str
    required_slots: list[dict[str, str]]
    missing_slots: list[dict[str, str]]
    introduced_specifics: list[str]
    commands: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-data-dir",
        default=str(Path(__file__).parent.parent / "clasharness" / "data"),
        help="Task data root containing tasks/ and datasets/ (default: clasharness/data)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default=None,
        help="Rewrite only tasks listed in this split file.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Rewrite only specific task_id (can pass multiple times).",
    )
    parser.add_argument(
        "--task-prefix",
        default="complex_",
        help="Rewrite tasks whose id starts with this prefix (default: complex_).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks processed (0 = no limit).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for sampling order when --limit > 0. "
            "If omitted, sampling order is randomized per run."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mock"],
        default="openai",
        help="Rewrite provider: openai or mock (default: openai).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="OpenAI API base URL (default: https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--openai-transport",
        choices=["auto", "urllib", "curl"],
        default="auto",
        help="HTTP transport for OpenAI requests (default: auto).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="OpenAI temperature (default: 0.2).",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=45,
        help="HTTP timeout in seconds for OpenAI calls (default: 45).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply rewritten question back into specs.json instruction field.",
    )
    parser.add_argument(
        "--store-as-variants",
        action="store_true",
        help=(
            "Store rewritten questions in instruction_variants while keeping the "
            "canonical instruction intact."
        ),
    )
    parser.add_argument(
        "--out-json",
        default="",
        help=(
            "Optional rewrite report JSON path. "
            "If omitted, no aggregate report file is written."
        ),
    )
    return parser.parse_args()


def _load_split_ids(task_data_dir: Path, split: str) -> set[str]:
    split_path = task_data_dir / "datasets" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return {
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _list_task_specs(task_data_dir: Path) -> list[Path]:
    tasks_dir = task_data_dir / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    return sorted(
        p / "specs.json"
        for p in tasks_dir.iterdir()
        if p.is_dir() and (p / "specs.json").exists()
    )


def _print_progress(
    *,
    done: int,
    total: int,
    ok: int,
    slot_mismatch: int,
    error: int,
    applied: int,
) -> None:
    if total <= 0:
        return
    width = 28
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = (
        f"\r[{bar}] {done}/{total}  "
        f"ok={ok} slot={slot_mismatch} err={error} applied={applied}"
    )
    sys.stdout.write(msg)
    if done >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _extract_slots(instruction: str, commands: list[str]) -> list[dict[str, str]]:
    instruction_lc = instruction.lower()
    joined = "\n".join(commands)
    slots: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(slot_type: str, value: str) -> None:
        v = value.strip()
        if not v:
            return
        # Require preserving slots that already appear in the original question.
        # This avoids forcing hidden command-only details (for example due dates)
        # back into the rewritten user phrasing.
        if v.lower() not in instruction_lc:
            return
        key = (slot_type, v.lower())
        if key in seen:
            return
        seen.add(key)
        slots.append({"type": slot_type, "value": v})

    for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", joined):
        add("email", email)
    for location in re.findall(r"--location '([^']+)'", joined):
        add("location", location)
    for tz in re.findall(r"--timezone ([^\s]+)", joined):
        add("timezone", tz.strip("'\""))
    for channel in re.findall(r"--channel ([^\s]+)", joined):
        add("channel", channel.strip("'\""))
    for target in re.findall(r"--target ([^\s]+)", joined):
        add("target", target.strip("'\""))
    for dt in re.findall(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?\b", joined):
        add("datetime", dt)
    for cron in re.findall(r"--cron '([^']+)'", joined):
        add("cron", cron)
    for query in re.findall(r"--query '([^']+)'", joined):
        add("query", query)

    return slots


def _missing_slots(new_question: str, slots: list[dict[str, str]]) -> list[dict[str, str]]:
    nq = new_question.lower()
    missing: list[dict[str, str]] = []
    for slot in slots:
        if slot["value"].lower() not in nq:
            missing.append(slot)
    return missing


def _extract_specific_literals(text: str) -> set[str]:
    vals: set[str] = set()

    for dt in re.findall(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?\b", text):
        vals.add(dt.lower())
    for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        vals.add(email.lower())
    for tz in re.findall(r"\b[A-Za-z]+/[A-Za-z0-9_+-]+\b", text):
        vals.add(tz.lower())
    for channel in re.findall(r"#[A-Za-z0-9_-]+", text):
        vals.add(channel.lower())
    for agent in re.findall(r"\bagent-[A-Za-z0-9_-]+\b", text):
        vals.add(agent.lower())
    for quoted in re.findall(r"'([^']+)'", text):
        q = quoted.strip()
        if q:
            vals.add(q.lower())
    for quoted in re.findall(r'"([^"]+)"', text):
        q = quoted.strip()
        if q:
            vals.add(q.lower())

    return vals


def _introduced_specifics(
    *,
    old_question: str,
    new_question: str,
    slots: list[dict[str, str]],
) -> list[str]:
    allowed = _extract_specific_literals(old_question)
    for slot in slots:
        allowed.add(slot["value"].lower())

    new_vals = _extract_specific_literals(new_question)
    introduced = sorted(v for v in new_vals if v not in allowed)
    return introduced


def _derive_compaction_hints(commands: list[str]) -> list[str]:
    hints: list[str] = []
    for cmd in commands:
        if cmd.startswith("tasks add "):
            m = re.search(r"--title '([^']+)'", cmd)
            if not m:
                continue
            title = m.group(1)
            tm = re.search(r"task for ([A-Za-z][A-Za-z\s-]+) \(([^)]+)\)", title, re.IGNORECASE)
            if tm:
                loc = tm.group(1).strip()
                topic = tm.group(2).strip()
                hints.append(
                    f"Do not copy the full generated task title verbatim; prefer concise wording like "
                    f"'follow-up task for {loc} about {topic}'."
                )
        if cmd.startswith("calendar add-event "):
            m = re.search(r"--title '([^']+)'", cmd)
            if not m:
                continue
            title = m.group(1)
            em = re.search(r"sync for ([A-Za-z][A-Za-z\\s-]+)$", title, re.IGNORECASE)
            if em:
                loc = em.group(1).strip()
                hints.append(
                    f"Do not copy long event title labels; concise wording like 'sync for {loc}' is preferred."
                )
    return hints


def _build_prompt(old_question: str, commands: list[str], slots: list[dict[str, str]]) -> str:
    compaction_hints = _derive_compaction_hints(commands)
    payload = {
        "old_question": old_question,
        "solution_commands": commands,
        "required_slots": slots,
        "compaction_hints": compaction_hints,
    }
    return textwrap.dedent(
        f"""\
        Rewrite this task question so it sounds like a natural real user request.
        Constraints:
        1) Keep the same task intent and hard constraints.
        2) Prefer outcome-focused phrasing; do NOT mirror the command checklist.
        3) Do NOT enumerate steps ("first/then/finally") and avoid procedural wording.
        4) Do NOT mention CLI, command names, or internal verification actions.
        5) Keep all required slots exactly.
        6) Do NOT introduce new specific entities (dates, IDs, locations, emails, channels, titles) beyond old_question + required_slots.
        7) Use one short natural sentence (prefer 10-24 words; hard max 30 words).
        8) Ignore scenario labels like "[...]" and do not copy them.
        9) Avoid synthetic/internal identifiers and template-like names (for example agent-xx, generated task titles) unless explicitly required.
        10) Prefer everyday phrasing a real user would type in chat.
        11) Use plain ASCII punctuation (no typographic quotes like “ ”).
        12) Avoid repeated words (for example "sync sync").
        13) Use direct request style (imperative), not polite-question style.
        14) Do NOT start with "Can you", "Could you", "Would you", "Please", or similar.
        15) Do NOT copy long synthetic task/event labels verbatim when concise semantic phrasing is enough.
        16) Output ONE English sentence only, no extra text.

        Style examples:
        - Better: "Check report-related emails and add a follow-up for Singapore."
        - Worse: "Can you check report-related emails and set up a follow-up for Singapore?"
        - Better: "Send a quick update to #general about the incident and track a follow-up task."
        - Worse: "Please send a quick update to #general and then create a task."

        Task payload:
        {json.dumps(payload, ensure_ascii=False)}
        """
    )


def _openai_rewrite(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: int,
    prompt: str,
    transport: str,
) -> str:
    if transport == "curl":
        return _openai_rewrite_curl(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            prompt=prompt,
        )
    if transport == "auto":
        try:
            return _openai_rewrite_urllib(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                timeout_s=timeout_s,
                prompt=prompt,
            )
        except (urllib.error.URLError, RuntimeError) as exc:
            msg = str(exc)
            if shutil.which("curl") and (
                "Network is unreachable" in msg
                or "Temporary failure in name resolution" in msg
                or "Name or service not known" in msg
            ):
                return _openai_rewrite_curl(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    prompt=prompt,
                )
            raise
    return _openai_rewrite_urllib(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        prompt=prompt,
    )


def _openai_rewrite_urllib(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: int,
    prompt: str,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rewrite benchmark tasks into natural chat-like user requests. "
                    "Keep intent and required entities, but avoid procedural checklist style. "
                    "Prefer direct imperative requests over polite questions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Some proxy gateways (for example Cloudflare worker front-ends)
            # may deny default Python urllib user-agents.
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        msg = f"HTTP Error {exc.code}"
        if detail:
            msg += f": {detail[:500]}"
        raise RuntimeError(msg) from exc
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response has no choices.")
    content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("OpenAI response content is empty.")
    return " ".join(content.split())


def _openai_rewrite_curl(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: int,
    prompt: str,
) -> str:
    if not shutil.which("curl"):
        raise RuntimeError("curl is not installed.")
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rewrite benchmark tasks into natural chat-like user requests. "
                    "Keep intent and required entities, but avoid procedural checklist style. "
                    "Prefer direct imperative requests over polite questions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            str(timeout_s),
            base_url.rstrip("/") + "/chat/completions",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "-H",
            "Accept: application/json",
            "-H",
            (
                "User-Agent: Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "--data-binary",
            "@-",
        ],
        input=json.dumps(body),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"curl transport failed (exit {proc.returncode}): {detail[:500]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"curl transport returned non-JSON response: {proc.stdout[:500]}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response has no choices.")
    content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("OpenAI response content is empty.")
    return " ".join(content.split())


def _cleanup_user_sentence(text: str) -> str:
    s = " ".join((text or "").split())
    # Normalize common smart punctuation to ASCII.
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("—", "-")
    )
    # Collapse duplicate adjacent words (case-insensitive), e.g. "sync sync".
    s = re.sub(r"\b([A-Za-z]+)(\s+\1\b)+", r"\1", s, flags=re.IGNORECASE)
    # Convert polite question openings into direct imperative style.
    s = re.sub(
        r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"^\s*(?:please|kindly)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*(?:i\s+need\s+you\s+to|i\s+want\s+you\s+to)\s+", "", s, flags=re.IGNORECASE)
    # Reduce template-like phrasing.
    s = re.sub(r"\b(task|event)\s+titled\s+", r"\1 ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bhigh-priority follow-up task\b", "high-priority task", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdelegated follow-up task\b", "follow-up task", s, flags=re.IGNORECASE)
    # Remove spaces before punctuation.
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = s.strip()
    if s.endswith("?"):
        s = s[:-1].rstrip() + "."
    elif s and s[-1] not in ".!?":
        s = s + "."
    if s:
        s = s[0].upper() + s[1:]
    return s.strip()


def _mock_rewrite(old_question: str) -> str:
    text = old_question.strip()
    text = re.sub(r"\[[^\]]+\]\s*", "", text)
    text = re.sub(r"\b(Then|then)\b", "and", text)
    text = re.sub(r"\b(finally|Finally)\b", "and", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("?"):
        text = text.rstrip(".") + "."
    return _cleanup_user_sentence(text)


def _rewrite_one(
    *,
    spec: dict[str, Any],
    provider: str,
    model: str,
    base_url: str,
    transport: str,
    temperature: float,
    timeout_s: int,
) -> tuple[str, str]:
    old_question = str(spec.get("old_question") or spec.get("instruction") or "").strip()
    commands = ((spec.get("ground_truth") or {}).get("solution_commands") or [])
    slots = _extract_slots(old_question, commands)
    prompt = _build_prompt(old_question, commands, slots)

    if provider == "mock":
        return _mock_rewrite(old_question), "mock"

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    rewritten = _openai_rewrite(
        api_key=api_key,
        base_url=base_url,
        model=model,
        transport=transport,
        temperature=temperature,
        timeout_s=timeout_s,
        prompt=prompt,
    )
    return _cleanup_user_sentence(rewritten), "openai"


def _apply_rewrite(
    spec: dict[str, Any],
    new_question: str,
    *,
    store_as_variants: bool,
    provider: str,
    model: str,
) -> None:
    old_question = str(spec.get("old_question") or spec.get("instruction") or "")
    if "old_question" not in spec:
        spec["old_question"] = old_question
    spec.setdefault("canonical_instruction", str(spec.get("instruction") or old_question))
    if store_as_variants:
        variants = list(spec.get("instruction_variants", []))
        if all(
            (
                item != new_question
                if isinstance(item, str)
                else str(item.get("text", "")).strip() != new_question
            )
            for item in variants
        ):
            variants.append(
                {
                    "style": "direct",
                    "text": new_question,
                    "provider": provider,
                    "model": model,
                }
            )
        spec["instruction_variants"] = variants
        return
    spec["instruction"] = new_question


def main() -> None:
    args = parse_args()
    task_data_dir = Path(args.task_data_dir).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else None
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)

    split_ids: set[str] | None = None
    if args.split:
        split_ids = _load_split_ids(task_data_dir, args.split)

    selected_task_ids = set(args.task_id or [])
    candidate_specs: list[Path] = []
    for spec_path in _list_task_specs(task_data_dir):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        task_id = str(spec.get("task_id") or "")
        if not task_id:
            continue
        if args.task_prefix and not task_id.startswith(args.task_prefix):
            continue
        if selected_task_ids and task_id not in selected_task_ids:
            continue
        if split_ids is not None and task_id not in split_ids:
            continue

        candidate_specs.append(spec_path)

    if args.limit > 0:
        if args.sample_seed is not None:
            rnd = random.Random(args.sample_seed)
            rnd.shuffle(candidate_specs)
        else:
            random.shuffle(candidate_specs)
        selected_specs = candidate_specs[: args.limit]
    else:
        selected_specs = candidate_specs

    results: list[RewriteResult] = []
    applied = 0
    total = len(selected_specs)
    ok_count = 0
    slot_count = 0
    err_count = 0

    for idx, spec_path in enumerate(selected_specs, start=1):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        task_id = str(spec.get("task_id") or "")
        if not task_id:
            continue

        old_question = str(spec.get("old_question") or spec.get("instruction") or "").strip()
        commands = ((spec.get("ground_truth") or {}).get("solution_commands") or [])
        slots = _extract_slots(old_question, commands)

        try:
            new_question, used_provider = _rewrite_one(
                spec=spec,
                provider=args.provider,
                model=args.model,
                base_url=args.openai_base_url,
                transport=args.openai_transport,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
            )
            missing = _missing_slots(new_question, slots)
            introduced = _introduced_specifics(
                old_question=old_question,
                new_question=new_question,
                slots=slots,
            )
            if missing:
                status = "slot_mismatch"
                reason = "rewritten question missing required slot(s)"
            else:
                status = "ok"
                reason = ""
        except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            new_question = old_question
            used_provider = args.provider
            missing = []
            introduced = []
            status = "error"
            reason = str(exc)

        if status == "ok" and args.apply and new_question != old_question:
            _apply_rewrite(
                spec,
                new_question,
                store_as_variants=bool(args.store_as_variants),
                provider=used_provider,
                model=args.model,
            )
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
            applied += 1

        results.append(
            RewriteResult(
                task_id=task_id,
                spec_path=str(spec_path),
                old_question=old_question,
                new_question=new_question,
                status=status,
                reason=reason,
                required_slots=slots,
                missing_slots=missing,
                introduced_specifics=introduced,
                commands=commands,
            )
        )
        if status == "ok":
            ok_count += 1
        elif status == "slot_mismatch":
            slot_count += 1
        else:
            err_count += 1
        _print_progress(
            done=idx,
            total=total,
            ok=ok_count,
            slot_mismatch=slot_count,
            error=err_count,
            applied=applied,
        )

    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "slot_mismatch": sum(1 for r in results if r.status == "slot_mismatch"),
        "error": sum(1 for r in results if r.status == "error"),
        "applied": applied,
    }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "task_data_dir": str(task_data_dir),
        "provider": args.provider,
        "model": args.model,
        "openai_transport": args.openai_transport,
        "split": args.split,
        "task_prefix": args.task_prefix,
        "applied": bool(args.apply),
        "summary": summary,
        "rewrites": [r.__dict__ for r in results],
    }
    print(f"Rewrites processed: {summary['total']}")
    print("OK: {ok} | slot_mismatch: {slot_mismatch} | error: {error}".format(**summary))
    print(f"Applied to specs: {summary['applied']}")
    if summary["error"] > 0:
        reasons = Counter(
            r.reason.strip() or "(empty reason)"
            for r in results
            if r.status == "error"
        )
        print("Top error reasons:")
        for reason, count in reasons.most_common(5):
            print(f"  - {count}x {reason}")
    if out_json is not None:
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"Output JSON: {out_json}")


if __name__ == "__main__":
    sys.exit(main())
