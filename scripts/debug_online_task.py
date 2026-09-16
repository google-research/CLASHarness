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

"""Run one task episode and save trajectory JSONL for mentor/debug review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.train_and_eval import ExpertAgent, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one task in online mode and save a debug trajectory JSONL."
    )
    parser.add_argument(
        "--task-id",
        default="agent_create_and_configure_4",
        help="Task ID to run (default: agent_create_and_configure_4)",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "multi", "real", "hybrid"],
        default="hybrid",
        help="Backend mode (default: hybrid)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/trajectories/debug_online",
        help="Directory to write JSONL (default: data/trajectories/debug_online)",
    )
    parser.add_argument(
        "--task-data-dir",
        default=str(
            Path(__file__).parent.parent
            / "clasharness"
            / "data"
        ),
        help=(
            "Task data root containing tasks/ and datasets/. "
            "Default: clasharness/data"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename (default: <task-id>.jsonl)",
    )
    parser.add_argument(
        "--online-clean",
        action="store_true",
        help="Store cleaned stdout/stderr fields in trajectory",
    )
    parser.add_argument(
        "--fallback-openclaw-network-to-mock",
        action="store_true",
        help="Fallback openclaw commands to mock on network/auth/dependency failures",
    )
    parser.add_argument(
        "--strict-online-data",
        action="store_true",
        help=(
            "In real/hybrid mode, disable mock fallback for weather/calendar/email/tasks "
            "online reads; fail when live data is unavailable (default: on)"
        ),
    )
    parser.add_argument(
        "--no-strict-online-data",
        dest="strict_online_data",
        action="store_false",
        help="Allow weather/calendar/email/tasks mock fallback when live fetch is unavailable.",
    )
    parser.set_defaults(strict_online_data=True)
    parser.add_argument(
        "--skip-incompatible-openclaw",
        dest="skip_incompatible_openclaw",
        action="store_true",
        default=True,
        help="Skip incompatible real openclaw commands (default: on)",
    )
    parser.add_argument(
        "--no-skip-incompatible-openclaw",
        dest="skip_incompatible_openclaw",
        action="store_false",
        help="Do not skip incompatible real openclaw commands",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-step details",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backend_kwargs = None
    if args.mode in {"real", "hybrid"}:
        backend_kwargs = {
            "skip_incompatible_openclaw": args.skip_incompatible_openclaw,
            "fallback_openclaw_network_to_mock": args.fallback_openclaw_network_to_mock,
            "strict_online_data": args.strict_online_data,
        }

    result = run_episode(
        task_id=args.task_id,
        agent=ExpertAgent(),
        task_data_dir=args.task_data_dir,
        mode=args.mode,
        record_trajectory=True,
        online_clean=args.online_clean,
        backend_kwargs=backend_kwargs,
        verbose=args.verbose,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = args.output_name or f"{args.task_id}.jsonl"
    out_file = out_dir / filename
    out_file.write_text(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    print("Saved:")
    print(f"  {out_file}")
    print("Summary:")
    print(f"  task_id   : {result.task_id}")
    print(f"  success   : {result.success}")
    print(f"  score     : {result.score:.2f}")
    print(f"  steps     : {result.steps}")
    print(f"  duration  : {result.duration_s:.2f}s")
    print(f"  final_rsp : {result.final_response or '(none)'}")
    print(f"  error     : {result.error or '(none)'}")

    if result.trajectory:
        print("Trajectory:")
        for step in result.trajectory:
            stdout_preview = (step.get("stdout") or "").strip().replace("\n", " | ")
            stderr_preview = (step.get("stderr") or "").strip().replace("\n", " | ")
            execution_trace = step.get("execution_trace")
            print(f"  step      : {step.get('step')}")
            print(f"  action    : {step.get('action')}")
            print(f"  exit_code : {step.get('exit_code')}")
            print(f"  reward    : {step.get('reward')}")
            print(f"  done      : {step.get('done')}")
            print(f"  stdout    : {stdout_preview[:200] or '(empty)'}")
            print(f"  stderr    : {stderr_preview[:200] or '(empty)'}")
            if isinstance(execution_trace, list) and execution_trace:
                print("  exec_path :")
                for idx, trace in enumerate(execution_trace, 1):
                    if not isinstance(trace, dict):
                        print(f"    [{idx}] {trace}")
                        continue
                    trace_action = trace.get("action") or "(unknown)"
                    trace_provider = trace.get("provider") or "-"
                    trace_exit = trace.get("exit_code")
                    trace_reason = trace.get("reason")
                    trace_request = trace.get("request")
                    trace_replay_cmd = trace.get("replay_cmd")
                    trace_stdout = str(trace.get("stdout") or "").strip().replace("\n", " | ")
                    trace_stderr = str(trace.get("stderr") or "").strip().replace("\n", " | ")
                    if trace_reason:
                        print(
                            f"    [{idx}] action={trace_action} provider={trace_provider} "
                            f"exit={trace_exit} reason={trace_reason}"
                        )
                    else:
                        print(
                            f"    [{idx}] action={trace_action} provider={trace_provider} "
                            f"exit={trace_exit}"
                        )
                    if isinstance(trace_request, dict) and trace_request:
                        request_preview = json.dumps(
                            trace_request, ensure_ascii=False
                        ).replace("\n", " | ")
                        print(f"         request={request_preview[:240]}")
                    if isinstance(trace_replay_cmd, str) and trace_replay_cmd.strip():
                        print(f"         replay_cmd={trace_replay_cmd[:360]}")
                    if trace_stdout:
                        print(f"         stdout={trace_stdout[:240]}")
                    if trace_stderr:
                        print(f"         stderr={trace_stderr[:240]}")
            print("  ---")


if __name__ == "__main__":
    main()
