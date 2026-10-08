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

"""OpenClaw Evaluation Results Visualizer Compiler.

Parses JSONL trajectories and parallel_run.summary from an evaluation run
directory and compiles an interactive visualize.html dashboard.
"""

import os
import sys
import glob
import json
import re
from pathlib import Path
import argparse


def _repo_root() -> Path:
  """Directory holding ``clasharness/`` -- the checkout root.

  Walks up from this file looking for ``pyproject.toml``. ``CLASHARNESS_REPO_ROOT``
  overrides it for callers that run the script from outside the checkout.
  """
  override = os.environ.get("CLASHARNESS_REPO_ROOT")
  if override:
    return Path(override)
  for parent in Path(__file__).resolve().parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  return Path(".")



def parse_summary_file(summary_path: Path) -> dict:
  """Parses key stats from parallel_run.summary using regex."""
  stats = {}
  if not summary_path.is_file():
    return stats

  try:
    content = summary_path.read_text(encoding="utf-8")
    
    # Extract basic configs
    m_config = re.search(r"Agent\s*:\s*([^\s]+)\s+Split\s*:\s*([^\s]+)\s+Mode\s*:\s*([^\s]+)", content)
    if m_config:
      stats["agent"] = m_config.group(1)
      stats["split"] = m_config.group(2)
      stats["mode"] = m_config.group(3)
    
    m_llm = re.search(r"LLM\s*:\s*([^\n]+)", content)
    if m_llm:
      stats["model"] = m_llm.group(1).strip()

    # Extract Tasks stats
    m_tasks = re.search(r"Tasks\s*:\s*(\d+)/(\d+)\s+passed\s+pass@budget=([^\s]+)\s+raw_accuracy=([^\s]+)\s+avg_score=([^\s]+)", content)
    if m_tasks:
      stats["passed_tasks"] = int(m_tasks.group(1))
      stats["total_tasks"] = int(m_tasks.group(2))
      stats["pass_rate"] = m_tasks.group(4)
      stats["avg_score"] = float(m_tasks.group(5))

    # Extract Completed stats
    m_completed = re.search(r"Completed\s*:\s*(\d+)/(\d+)\s+provider_failures=(\d+)\s+provider_impacted_tasks=(\d+)\s+clean_completed_tasks=(\d+)\s+provider_adjusted_accuracy=([^\s]+)", content)
    if m_completed:
      stats["completed_tasks"] = int(m_completed.group(5))
      stats["provider_failures"] = int(m_completed.group(3))
      stats["adjusted_accuracy"] = m_completed.group(6)

    # Extract Diagnostics
    m_diag = re.search(r"Diagnostics\s*:\s*near_miss_rate=([^\s]+)\s+near_miss_tasks=(\d+)", content)
    if m_diag:
      stats["near_miss_rate"] = m_diag.group(1)
      stats["near_miss_tasks"] = int(m_diag.group(2))

  except Exception as e:
    print(f"Warning: Failed to parse summary file: {e}")
  
  return stats


def backfill_run_metadata(runs: list[dict]) -> None:
  """Backfills missing metadata (such as scenario summary and instruction) from workspace spec files."""
  repo_root = _repo_root()

  for run in runs:
    if "metadata" not in run:
      run["metadata"] = {}
    
    metadata = run["metadata"]
    instruction = metadata.get("instruction") or run.get("task_instruction")
    summary = metadata.get("scenario_summary")
    
    slug = metadata.get("scenario_slug") or run.get("scenario_slug")
    if not slug:
      task_id = run.get("task_id", "")
      match = re.match(r"conflict_vlm_(.+)_r\d+", task_id)
      if match:
        slug = match.group(1)
        metadata["scenario_slug"] = slug
    
    if (not instruction or not summary) and slug:
      # Search in data/scenarios/*/slug/slug_spec.json
      spec_files = list(repo_root.glob(f"clasharness/data/scenarios/*/{slug}/{slug}_spec.json"))
      if spec_files:
        spec_file = spec_files[0]
        try:
          with open(spec_file, "r", encoding="utf-8") as f:
            spec = json.load(f)
            if not summary:
              summary = spec.get("metadata", {}).get("summary", "")
            if not instruction:
              round_id = metadata.get("round_id") or run.get("round_id")
              if not round_id:
                task_id = run.get("task_id", "")
                rm = re.search(r"_(r\d+)$", task_id)
                if rm:
                  round_id = rm.group(1)
              
              if round_id:
                for r_obj in spec.get("rounds", []):
                  if r_obj.get("round_id") == round_id:
                    instruction = r_obj.get("instruction", "")
                    break
        except Exception:
          pass
    
    # Save back to metadata
    metadata["instruction"] = instruction or ""
    metadata["scenario_summary"] = summary or ""


def _load_asset_paths(runs: list[dict], out_dir: Path) -> dict:
  """Builds a map of image asset references (relative file paths) for the
  scenarios in the runs.

  Previously this base64-encoded every image into a data: URI and inlined it
  into visualize.html; aggregating many scenarios/seeds pushed the payload past
  the filesystem's max file size (EFBIG: [Errno 27] File too large). Instead we
  emit a path relative to the dashboard's own directory (out_dir) so the browser
  loads each image lazily from the in-place scenario assets when the report is
  opened. The de-duplicated {blobs, alias} layout is understood by
  resolveAssetBlob() in the HTML template.
  """
  assets_map = {}
  
  # Find unique scenarios
  scenarios = set()
  for run in runs:
    metadata = run.get("metadata", {})
    gen_id = metadata.get("generator_id") or "conflict_vlm_v1"
    slug = metadata.get("scenario_slug") or run.get("scenario_slug")
    if slug:
      scenarios.add((gen_id, slug))
      
  # The checkout root first, then the directory above this file in case
  # pyproject.toml is missing and _repo_root() fell back to the cwd.
  scenario_roots = []
  for root in (_repo_root(), Path(__file__).resolve().parents[1]):
    if root not in scenario_roots:
      scenario_roots.append(root)

  try:
    out_dir_abs = Path(out_dir).resolve()
  except Exception:
    out_dir_abs = Path(out_dir)

  def _ref_for(asset_file: Path) -> str:
    """Path to embed in <img src>, relative to out_dir when possible."""
    resolved = asset_file.resolve()
    try:
      return Path(os.path.relpath(resolved, out_dir_abs)).as_posix()
    except Exception:
      return resolved.as_posix()

  for gen_id, slug in scenarios:
    candidates = (root / "clasharness" / "data" / "scenarios" / gen_id / slug
                  for root in scenario_roots)
    scenario_path = next((p for p in candidates if p.is_dir()), None)
    if scenario_path is None:
      continue
      
    manifest_path = scenario_path / "asset_manifest.json"
    if not manifest_path.is_file():
      continue
      
    try:
      with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
      
      assets_list = manifest.get("assets", manifest if isinstance(manifest, list) else [])
      # De-duplicated layout: each unique image is referenced once under "blobs"
      # (keyed by asset_id); "alias" maps the alternative lookup keys (relative
      # path, file name) to the owning asset_id.
      blobs = {}
      alias = {}
      for asset in assets_list:
        asset_id = asset.get("id")
        rel_path = asset.get("path")
        if not asset_id or not rel_path:
          continue
          
        asset_file = scenario_path / "assets" / rel_path
        if asset_file.is_file():
          blobs[asset_id] = _ref_for(asset_file)
          alias[rel_path] = asset_id
          alias[Path(rel_path).name] = asset_id
          
      assets_map[slug] = {"blobs": blobs, "alias": alias}
    except Exception as e:
      print(f"Warning: Failed to load assets for {slug}: {e}")
      
  return assets_map


def parse_args():
  parser = argparse.ArgumentParser(
      description="Compile an interactive evaluation-results visualize.html dashboard."
  )
  parser.add_argument(
      "--eval_dir", "--eval-dir", dest="eval_dir", required=True,
      help="Path to the evaluation results folder containing JSONL files.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  eval_dir = args.eval_dir
  if not os.path.isabs(eval_dir):
    root_override = os.environ.get("CLASHARNESS_REPO_ROOT")
    if root_override:
      eval_path = (Path(root_override) / eval_dir).resolve()
    else:
      eval_path = Path(eval_dir).resolve()
  else:
    eval_path = Path(eval_dir)

  if not eval_path.is_dir():
    print(f"Error: Evaluation directory does not exist: {eval_path}")
    sys.exit(1)

  # Find all .jsonl files
  jsonl_files = sorted(eval_path.glob("*.jsonl"))
  if not jsonl_files:
    print(f"Error: Could not find any .jsonl trajectory files in {eval_path}")
    sys.exit(1)

  print(f"Found {len(jsonl_files)} trajectory file(s).")
  
  # Read trajectories
  runs = []
  for file_path in jsonl_files:
    print(f"Reading: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
      for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
          continue
        try:
          run_obj = json.loads(line)
          runs.append(run_obj)
        except Exception as e:
          print(f"  Warning: Failed to parse line {line_num} in {file_path.name}: {e}")

  print(f"Successfully loaded {len(runs)} task runs.")

  # Backfill missing metadata (summary, instruction) from spec files
  backfill_run_metadata(runs)

  # Load & parse parallel_run.summary
  summary_file = eval_path / "parallel_run.summary"
  summary_stats = parse_summary_file(summary_file)

  # Compute fallback stats if summary is missing/incomplete
  if "total_tasks" not in summary_stats:
    summary_stats["total_tasks"] = len(runs)
    passed = sum(1 for r in runs if r.get("success"))
    summary_stats["passed_tasks"] = passed
    summary_stats["pass_rate"] = f"{(passed / len(runs) * 100):.1f}%" if runs else "0.0%"
    summary_stats["avg_score"] = sum(r.get("score", 0.0) for r in runs) / len(runs) if runs else 0.0
    summary_stats["completed_tasks"] = len(runs)
    
    prov_fail = sum(1 for r in runs if r.get("provider_impacted") or r.get("error_type") == "provider_failure")
    summary_stats["provider_failures"] = prov_fail
    
    clean_runs = len(runs) - prov_fail
    summary_stats["adjusted_accuracy"] = f"{(passed / clean_runs * 100):.1f}%" if clean_runs > 0 else "0.0%"
    
    near_miss = sum(1 for r in runs if not r.get("success") and r.get("score", 0.0) > 0.0)
    summary_stats["near_miss_tasks"] = near_miss
    summary_stats["near_miss_rate"] = f"{(near_miss / len(runs) * 100):.1f}%" if runs else "0.0%"

  # Read visualize HTML template
  script_dir = Path(__file__).parent.resolve()
  template_path = script_dir / "templates" / "visualize_eval_template.html"
  if not template_path.exists():
    print(f"Error: Visualizer HTML template not found at {template_path}")
    sys.exit(1)

  with open(template_path, "r", encoding="utf-8") as f:
    template_content = f.read()

  # Reference image assets by path (relative to this dashboard) instead of
  # inlining base64 data URIs. Images load lazily from the in-place scenario
  # assets, keeping visualize.html small.
  assets_map = _load_asset_paths(runs, eval_path)

  # Package payload and render. The asset-free fallback below is a defensive
  # safety net; with path references (rather than inlined base64) the payload
  # no longer approaches the filesystem's per-file size cap (EFBIG).
  def _render(include_assets: bool) -> str:
    payload = {
        "summary": summary_stats,
        "runs": runs,
        "assets_map": assets_map if include_assets else {},
    }
    payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
    return template_content.replace("/* EVAL_DATA_PLACEHOLDER */", payload_str)

  # Save visualize.html
  output_file = eval_path / "visualize.html"
  output_content = _render(include_assets=True)
  try:
    with open(output_file, "w", encoding="utf-8") as f:
      f.write(output_content)
  except OSError as write_err:
    size_mb = len(output_content.encode("utf-8")) / 1e6
    print(
        f"Warning: writing visualize.html failed ({write_err}); payload was "
        f"~{size_mb:.0f} MB. Retrying without inline image assets."
    )
    output_content = _render(include_assets=False)
    with open(output_file, "w", encoding="utf-8") as f:
      f.write(output_content)
    print("Note: wrote visualize.html WITHOUT inline images (too large to embed).")

  print("\n======================================================================")
  print("SUCCESS: Standalone Evaluation Results Visualizer Compiled!")
  print(f"Open visualizer in browser: file://{output_file}")
  print("======================================================================\n")

if __name__ == "__main__":
  main()
