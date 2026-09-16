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

"""Generate LLM-backed CLASHarness PRO scenarios and compiled task specs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clasharness.scenario.llm_client import (  # noqa: E402
    MockScenarioLLMClient,
    OpenAICompatibleScenarioLLMClient,
    ScenarioLLMError,
)
from clasharness.scenario.llm_pipeline import LLMScenarioPipeline, load_seed_file  # noqa: E402
from clasharness.scenario.materializer import materialize_generated_scenario  # noqa: E402
from clasharness.scenario.spec import ScenarioSpec  # noqa: E402
from clasharness.scenario.strict_validator import strict_validate_scenario_package  # noqa: E402
from clasharness.core.task import Task, save_task  # noqa: E402

_FILE_TOKEN_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._\-]*\."
    r"(?:md|markdown|jsonl|eml|log|csv|tsv|txt|json|yaml|yml|pdf|docx?|xlsx?|ics)\b",
    re.IGNORECASE,
)
_ROUND_REF_PATTERNS = (
    re.compile(r"\bQ(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bR(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bROUND[_\- ]?(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bR[_\- ]?(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bQ[_\- ]?(\d{1,2})\b", re.IGNORECASE),
)
_SYSTEM_PRELOAD_FILES = {"emails.json", "tasks.json", "calendar_events.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen LLM-backed PRO scenario packages and runnable task specs."
    )
    parser.add_argument("--provider", choices=["openai", "mock"], default="openai")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--api",
        choices=["auto", "chat", "responses"],
        default="auto",
        help="OpenAI-compatible API shape to use (default: auto).",
    )
    parser.add_argument(
        "--transport",
        choices=["urllib", "curl"],
        default="urllib",
        help="HTTP transport. Use curl when a proxy blocks Python urllib.",
    )
    parser.add_argument(
        "--json-mode",
        choices=["none", "response_format"],
        default="none",
        help="Use OpenAI response_format JSON mode for chat APIs. Keep 'none' for Claude/LiteLLM.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Per-layer provider request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Provider retries per layer after retryable failures.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Sleep before each provider request to avoid local proxy/provider rate limits.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Provider output-token cap per generated layer.",
    )
    parser.add_argument(
        "--raw-response-dir",
        default=None,
        help="Optional directory for raw provider JSON responses, useful when JSON extraction fails.",
    )
    parser.add_argument("--seed-dir", default="clasharness/data/scenario_seeds/pro_llm_v1")
    parser.add_argument(
        "--seed-slug",
        action="append",
        default=None,
        help="Only generate the given seed slug. Can be passed multiple times.",
    )
    parser.add_argument("--output-scenario-dir", default="clasharness/data/scenarios/pro_llm_v1")
    parser.add_argument("--output-data-dir", default="/tmp/clasharness-llm-data")
    parser.add_argument("--target-rounds", type=int, default=None)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Do not clear the compiled task output dir first. The scenario package "
            "for each generated seed is still refreshed to avoid stale artifacts."
        ),
    )
    parser.add_argument(
        "--compile-existing-scenarios",
        action="store_true",
        help=(
            "Do not call the LLM. Compile existing scenario.json packages under "
            "--output-scenario-dir into --output-data-dir."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print scenario-level summary, not layer/request progress.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_dir).expanduser()
    output_scenario_dir = Path(args.output_scenario_dir).expanduser()
    output_data_dir = Path(args.output_data_dir).expanduser()

    seed_paths = sorted(seed_dir.glob("*.json"))
    if args.seed_slug:
        wanted = set(args.seed_slug)
        seed_paths = [path for path in seed_paths if path.stem in wanted]
        missing = sorted(wanted - {path.stem for path in seed_paths})
        if missing:
            raise SystemExit(f"Requested seed slug(s) not found under {seed_dir}: {', '.join(missing)}")
    if not seed_paths:
        raise SystemExit(f"No seed JSON files found under {seed_dir}")

    _log(f"Seed dir: {seed_dir}")
    _log(f"Found {len(seed_paths)} seed file(s)")
    _log(f"Output scenario dir: {output_scenario_dir}")
    _log(f"Output task data dir: {output_data_dir}")

    if not args.no_clean and not args.compile_existing_scenarios:
        if output_data_dir.exists():
            _log(f"Cleaning output task data dir: {output_data_dir}")
            shutil.rmtree(output_data_dir)
        output_data_dir.mkdir(parents=True, exist_ok=True)

    if args.compile_existing_scenarios:
        specs = _compile_existing_scenarios(
            seed_paths=seed_paths,
            output_scenario_dir=output_scenario_dir,
            output_data_dir=output_data_dir,
            clean=not args.no_clean,
        )
        _log(f"Compiled {len(specs)} existing scenario package(s)")
        _log(f"Compiled task data: {output_data_dir}")
        _log(f"Scenario packages: {output_scenario_dir}")
        return

    if args.provider == "mock":
        client = MockScenarioLLMClient()
    else:
        client = OpenAICompatibleScenarioLLMClient(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            request_delay_seconds=args.request_delay_seconds,
            max_output_tokens=args.max_output_tokens,
            api=args.api,
            transport=args.transport,
            json_mode=args.json_mode,
            verbose=not args.quiet,
            logger=_log,
            raw_response_dir=Path(args.raw_response_dir).expanduser() if args.raw_response_dir else None,
        )
    pipeline = LLMScenarioPipeline(
        client,
        progress=None if args.quiet else _log,
    )

    specs = []
    for seed_path in seed_paths:
        seed = load_seed_file(seed_path)
        if args.target_rounds is not None:
            seed.target_rounds = args.target_rounds
            seed.validate()
        _log(f"Generating PRO LLM scenario: {seed.slug} ({seed.target_rounds} rounds)")
        try:
            layers = pipeline.generate(seed)
        except ScenarioLLMError as exc:
            raise SystemExit(f"LLM generation failed for {seed_path}: {exc}") from exc
        _log(f"Materializing package and compiled tasks for {seed.slug}")
        spec = materialize_generated_scenario(
            seed,
            layers,
            scenario_series_dir=output_scenario_dir,
            output_data_dir=output_data_dir,
            clean=True,
        )
        scenario_dir = output_scenario_dir / "pro_scenario_workflow" / seed.slug
        _log(f"Validating scenario package: {scenario_dir}")
        validation = strict_validate_scenario_package(spec, scenario_dir)
        validation.raise_for_errors()
        specs.append(spec)
        _log(
            f"Wrote {scenario_dir / 'scenario.json'} "
            f"({len(spec.rounds)} rounds, {len(spec.sessions)} sessions, "
            f"{len(spec.workspace.files)} workspace files, {len(spec.updates)} updates)"
        )

    _log(f"Generated {len(specs)} scenario package(s)")
    _log(f"Compiled task data: {output_data_dir}")
    _log(f"Scenario packages: {output_scenario_dir}")


def _compile_existing_scenarios(
    *,
    seed_paths: list[Path],
    output_scenario_dir: Path,
    output_data_dir: Path,
    clean: bool,
) -> list[ScenarioSpec]:
    specs_and_dirs: list[tuple[ScenarioSpec, Path, Path]] = []
    for seed_path in seed_paths:
        seed = load_seed_file(seed_path)
        scenario_dir = output_scenario_dir / "pro_scenario_workflow" / seed.slug
        scenario_json = scenario_dir / "scenario.json"
        if not scenario_json.exists():
            raise SystemExit(
                f"Missing existing scenario package for seed {seed.slug}: {scenario_json}"
            )
        spec = ScenarioSpec.read_json(scenario_json)
        from clasharness.scenario.consistency_layer import audit_and_repair_scenario
        spec = audit_and_repair_scenario(spec)
        sanitized = _sanitize_existing_compiled_task_specs(spec)
        if sanitized:
            _log(
                f"Sanitized {sanitized} legacy round task_spec(s) before validation "
                f"for {seed.slug}"
            )
        validation = strict_validate_scenario_package(spec, scenario_dir)
        if validation.errors:
            _log(f"Warning: Skipping {seed.slug} due to validation errors:\n" + "\n".join(f"  - {e}" for e in validation.errors))
            continue
        specs_and_dirs.append((spec, scenario_dir, scenario_json))

    if clean and output_data_dir.exists():
        _log(f"Cleaning output task data dir: {output_data_dir}")
        shutil.rmtree(output_data_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    specs: list[ScenarioSpec] = []
    for spec, scenario_dir, scenario_json in specs_and_dirs:
        for round_item in spec.rounds:
            if not isinstance(round_item.task_spec, dict) or not round_item.task_spec:
                raise SystemExit(
                    f"Scenario {spec.scenario_slug} round {round_item.round_id} has no compiled task_spec"
                )
            save_task(Task.from_dict(round_item.task_spec), data_dir=output_data_dir)
        specs.append(spec)
        _log(f"Compiled {len(spec.rounds)} task(s) from {scenario_json}")

    _write_splits(output_data_dir)
    return specs


def _sanitize_existing_compiled_task_specs(spec: ScenarioSpec) -> int:
    """Clean legacy embedded task specs before compile-existing validation.

    Older scenario packages can contain final/full workspace indexes inside
    round-visible task specs. Current generation prevents that, but
    compile-existing should be able to rebuild task data from already frozen
    packages without another LLM call.
    """

    changed = 0
    for round_item in spec.rounds:
        task_spec = round_item.task_spec
        if not isinstance(task_spec, dict):
            continue
        private = (
            task_spec.get("data", {})
            .get("private", {})
            if isinstance(task_spec.get("data"), dict)
            else {}
        )
        overrides = (
            private.get("initial_state_overrides", {})
            if isinstance(private, dict)
            else {}
        )
        if not isinstance(overrides, dict):
            continue
        round_index = _round_index(round_item.round_id)
        overrides = {
            key: value
            for key, value in overrides.items()
            if not _workspace_key_is_future_round_file(str(key), round_index)
        }
        private["initial_state_overrides"] = overrides
        visible_names = {
            Path(str(path)).name
            for path in overrides
            if str(path).startswith("workspace/")
        } | _SYSTEM_PRELOAD_FILES
        sanitized_overrides = _sanitize_runtime_value(
            overrides,
            round_index=round_index,
            visible_names=visible_names,
        )
        _restore_expected_anchors(
            sanitized_overrides,
            _expected_anchor_ids_for_round(round_item),
            round_index=round_index,
        )
        if sanitized_overrides != overrides:
            private["initial_state_overrides"] = sanitized_overrides
            changed += 1
    return changed


def _expected_anchor_ids_for_round(round_item) -> list[str]:
    anchors = [
        str(item)
        for item in getattr(round_item, "expected_anchor_ids", [])
        if isinstance(item, str) and item.strip()
    ]
    if anchors:
        return anchors
    task_spec = getattr(round_item, "task_spec", {}) or {}
    answer = (
        task_spec.get("ground_truth", {}).get("answer", {})
        if isinstance(task_spec, dict) and isinstance(task_spec.get("ground_truth"), dict)
        else {}
    )
    return [
        str(item)
        for item in answer.get("expected_anchor_ids", [])
        if isinstance(item, str) and item.strip()
    ]


def _restore_expected_anchors(
    overrides: dict,
    expected_anchor_ids: list[str],
    *,
    round_index: int,
) -> None:
    if not expected_anchor_ids:
        return
    current = json.dumps(overrides, ensure_ascii=False, sort_keys=True)
    missing = [
        anchor
        for anchor in expected_anchor_ids
        if anchor not in current and not (round_index < 8 and anchor.startswith(("DECISION:", "RULE:", "UPDATE:")))
    ]
    if not missing:
        return
    target_key = "workspace/index.md"
    if target_key not in overrides:
        workspace_keys = sorted(
            key for key in overrides if str(key).startswith("workspace/")
        )
        if not workspace_keys:
            return
        target_key = workspace_keys[0]
    current_text = str(overrides.get(target_key) or "")
    suffix = "\n\nEvidence anchors: " + ", ".join(missing) + "\n"
    overrides[target_key] = current_text.rstrip() + suffix


def _sanitize_runtime_value(value, *, round_index: int, visible_names: set[str]):
    if isinstance(value, dict):
        return {
            key: _sanitize_runtime_value(
                item,
                round_index=round_index,
                visible_names=visible_names,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_runtime_value(
                item,
                round_index=round_index,
                visible_names=visible_names,
            )
            for item in value
        ]
    if isinstance(value, str):
        kept = [
            line
            for line in value.splitlines()
            if not _line_references_invisible_file(line, visible_names)
            and not _line_references_future_round(line, round_index)
        ]
        if not kept:
            return ""
        return "\n".join(kept) + ("\n" if value.endswith("\n") else "")
    return value


def _line_references_invisible_file(line: str, visible_names: set[str]) -> bool:
    for token in _FILE_TOKEN_RE.findall(str(line or "")):
        if token not in visible_names:
            return True
    return False


def _line_references_future_round(line: str, round_index: int) -> bool:
    text = str(line or "")
    if "Total Rounds" in text or "Target Rounds" in text:
        return False
    for pattern in _ROUND_REF_PATTERNS:
        for match in pattern.finditer(text):
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            if round_index < value <= 24:
                return True
    return False


def _workspace_key_is_future_round_file(key: str, round_index: int) -> bool:
    if not key.startswith("workspace/"):
        return False
    hint = _round_hint_from_path(key)
    return bool(hint and round_index < hint <= 24)


def _round_hint_from_path(path: str) -> int:
    text = str(path or "")
    patterns = (
        r"(?:^|[._/\-])(?:q|r|round)[_\-]?(\d{1,2})(?:[._/\-]|$)",
        r"(?:^|[._/\-])round[_\- ]?(\d{1,2})(?:[._/\-]|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            if 1 <= value <= 24:
                return value
    return 0


def _round_index(round_id: str | None) -> int:
    text = str(round_id or "").lower()
    return int(text[1:]) if len(text) == 3 and text.startswith("q") and text[1:].isdigit() else 0


def _write_splits(output_data_dir: Path) -> None:
    datasets_dir = output_data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_data_dir / "tasks"
    task_ids = sorted(
        path.name
        for path in tasks_dir.iterdir()
        if path.is_dir() and (path / "specs.json").exists()
    )
    for split in ("train", "dev", "test", "total"):
        (datasets_dir / f"{split}.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
