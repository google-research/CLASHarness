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

"""Generate CLASHarness PRO-VLM scenario packages and task data."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import os
print("GEN_PRO_VLM_LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
for path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
  if os.path.exists(path) and os.path.isdir(path):
    files = [f for f in os.listdir(path) if "jpeg" in f or "jpg" in f]
    if files:
      print(f"Scan {path}: {files}")

from clasharness.core.task import Task, save_task
from clasharness.scenario.llm_client import OpenAICompatibleScenarioLLMClient, ScenarioLLMError
from clasharness.scenario.spec import ScenarioSpec
from clasharness.scenario.strict_validator import strict_validate_scenario_package
from clasharness.scenario.vlm_materializer import load_vlm_seed_file, materialize_vlm_scenario
from clasharness.scenario.vlm_spec import VLMScenarioSpec, validate_vlm_scenario_spec


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "openai"],
        help="Text/spec/caption provider. Use openai for Claude Code/LiteLLM OpenAI-compatible APIs.",
    )
    parser.add_argument("--model", default="gpt-4.1", help="Text/spec/caption model for --provider openai.")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="OpenAI-compatible base URL for text/spec/caption generation.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key env var for text/spec/caption generation.")
    parser.add_argument("--api", choices=["auto", "chat", "responses"], default="auto")
    parser.add_argument("--transport", choices=["urllib", "curl"], default="urllib")
    parser.add_argument("--json-mode", choices=["none", "response_format"], default="none")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed-dir", default="clasharness/data/scenario_seeds/pro_vlm_v1")
    parser.add_argument("--seed-slug", action="append", default=None, help="Generate only this seed slug. Can be repeated.")
    parser.add_argument("--output-scenario-dir", default="clasharness/data/scenarios/pro_vlm_v1")
    parser.add_argument("--output-data-dir", default="/tmp/clasharness-vlm-data")
    parser.add_argument("--target-rounds", type=int, default=24)
    parser.add_argument("--compile-existing-scenarios", action="store_true", help="Compile existing scenario.json task specs into output-data-dir.")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean output dirs before writing.")
    parser.add_argument("--image-provider", default="mock", choices=["mock", "google"], help="Provider for natural-image assets.")
    parser.add_argument("--chart-provider", default="template", choices=["template", "code"], help="Provider for charts/tables/dashboards.")
    parser.add_argument("--google-model", default="gemini-3.1-flash-image", help="Google AI Studio Gemini image model.")
    parser.add_argument("--google-api-key-env", default="GEMINI_API_KEY", help="Environment variable holding the Google AI Studio API key.")
    parser.add_argument("--google-base-url", default="https://generativelanguage.googleapis.com/v1", help="Google Gemini API base URL.")
    parser.add_argument("--raw-response-dir", default=None, help="Optional ignored/local directory for raw text/image provider responses.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_dir)
    if not seed_dir.is_absolute():
        seed_dir = ROOT / seed_dir
    scenario_series_dir = Path(args.output_scenario_dir)
    if not scenario_series_dir.is_absolute():
        scenario_series_dir = ROOT / scenario_series_dir
    output_data_dir = Path(args.output_data_dir)
    if not output_data_dir.is_absolute():
        output_data_dir = ROOT / output_data_dir
    clean = not args.no_clean

    if clean and output_data_dir.exists():
        shutil.rmtree(output_data_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    if args.compile_existing_scenarios:
        count = _compile_existing(scenario_series_dir, output_data_dir)
        print(f"Compiled {count} existing PRO-VLM scenario package(s)")
        print(f"Compiled task data: {output_data_dir}")
        return

    seed_files = sorted(seed_dir.glob("*.json"))
    if args.seed_slug:
        wanted = set(args.seed_slug)
        seed_files = [path for path in seed_files if path.stem in wanted]
        missing = sorted(wanted - {path.stem for path in seed_files})
        if missing:
            raise SystemExit(f"Requested VLM seed slug(s) not found under {seed_dir}: {', '.join(missing)}")
    if not seed_files:
        raise SystemExit(f"No VLM seed files found under {seed_dir}")

    raw_response_dir = Path(args.raw_response_dir) if args.raw_response_dir else None
    if raw_response_dir and not raw_response_dir.is_absolute():
        raw_response_dir = ROOT / raw_response_dir
    if args.provider == "mock":
        text_client = None
    else:
        text_client = OpenAICompatibleScenarioLLMClient(
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
            raw_response_dir=raw_response_dir,
        )

    generated = 0
    for seed_file in seed_files:
        seed = load_vlm_seed_file(seed_file)
        if args.target_rounds:
            seed = seed.__class__(
                slug=seed.slug,
                domain=seed.domain,
                title=seed.title,
                target_rounds=args.target_rounds,
                core_conflict=seed.core_conflict,
                participants=seed.participants,
                required_constraints=seed.required_constraints,
                allowed_tool_families=seed.allowed_tool_families,
                modalities=seed.modalities,
                generation_policy=seed.generation_policy,
                input_policy=seed.input_policy,
            )
        print(f"Generating PRO-VLM scenario: {seed.slug} ({seed.target_rounds} rounds)")
        print(f"  text/spec provider={args.provider} image_provider={args.image_provider} chart_provider={args.chart_provider}")
        try:
            spec = materialize_vlm_scenario(
                seed,
                scenario_series_dir=scenario_series_dir,
                output_data_dir=output_data_dir,
                clean=clean,
                provider=args.provider,
                image_provider=args.image_provider,
                chart_provider=args.chart_provider,
                google_model=args.google_model,
                google_api_key_env=args.google_api_key_env,
                google_base_url=args.google_base_url,
                raw_response_dir=raw_response_dir,
                llm_client=text_client,
                progress=None if args.quiet else _log,
            )
        except ScenarioLLMError as exc:
            raise SystemExit(f"VLM LLM generation failed for {seed_file}: {exc}") from exc
        scenario_dir = scenario_series_dir / "pro_vlm_workflow" / seed.slug
        validation = strict_validate_scenario_package(spec, scenario_dir)
        validation.raise_for_errors()
        vlm_spec_path = scenario_dir / "vlm_spec.json"
        if vlm_spec_path.exists():
            vlm_errors = validate_vlm_scenario_spec(VLMScenarioSpec.read_json(vlm_spec_path))
            if vlm_errors:
                raise ValueError("VLM spec validation failed:\n" + "\n".join(vlm_errors))
        print(
            f"Wrote {scenario_dir / 'scenario.json'} "
            f"({len(spec.rounds)} rounds, {len(spec.workspace.files)} workspace files)"
        )
        generated += 1

    print(f"Generated {generated} PRO-VLM scenario package(s)")
    print(f"Compiled task data: {output_data_dir}")
    print(f"Scenario packages: {scenario_series_dir}")


def _compile_existing(scenario_series_dir: Path, output_data_dir: Path) -> int:
    specs = sorted(scenario_series_dir.glob("pro_vlm_workflow/*/scenario.json"))
    task_ids: list[str] = []
    for spec_path in specs:
        spec = ScenarioSpec.read_json(spec_path)
        validation = strict_validate_scenario_package(spec, spec_path.parent)
        validation.raise_for_errors()
        vlm_spec_path = spec_path.parent / "vlm_spec.json"
        if vlm_spec_path.exists():
            errors = validate_vlm_scenario_spec(VLMScenarioSpec.read_json(vlm_spec_path))
            if errors:
                raise ValueError(f"VLM spec validation failed for {vlm_spec_path}:\n" + "\n".join(errors))
        for round_item in spec.rounds:
            task = Task.from_dict(round_item.task_spec)
            save_task(task, data_dir=output_data_dir)
            task_ids.append(task.task_id)
    datasets_dir = output_data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    task_ids = sorted(set(task_ids))
    for split in ("train", "dev", "test", "total"):
        (datasets_dir / f"{split}.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    return len(specs)


if __name__ == "__main__":
    main()
