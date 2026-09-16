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

"""OpenClaw Scenario Visualizer Compiler.

Generates a standalone visualize.html in the target scenario directory by
embedding the spec JSON.
"""

import os
import sys
import glob
import json
from pathlib import Path
import argparse


def parse_args():
  parser = argparse.ArgumentParser(
      description="Compile a standalone scenario visualize.html from a scenario spec."
  )
  parser.add_argument(
      "--scenario_dir", "--scenario-dir", dest="scenario_dir", required=True,
      help="Path to the materialized scenario folder.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  scenario_path = Path(args.scenario_dir).resolve()
  if not scenario_path.is_dir():
    print(f"Error: Scenario directory does not exist: {scenario_path}")
    sys.exit(1)

  # Find spec.json
  spec_files = list(scenario_path.glob("*_spec.json"))
  if not spec_files:
    print(f"Error: Could not find any *_spec.json in {scenario_path}")
    sys.exit(1)

  spec_file = spec_files[0]
  print(f"Loading scenario spec: {spec_file.name}")
  with open(spec_file, "r", encoding="utf-8") as f:
    spec_data = json.load(f)

  # Audit and repair scenario consistency before embedding
  try:
    package_root = Path(__file__).parent.parent.resolve()
    if str(package_root) not in sys.path:
      sys.path.insert(0, str(package_root))
    from clasharness.scenario.spec import ScenarioSpec
    from clasharness.scenario.consistency_layer import audit_and_repair_scenario
    spec_obj = ScenarioSpec.from_dict(spec_data)
    spec_obj = audit_and_repair_scenario(spec_obj)
    spec_data = spec_obj.to_dict()
    print("Consistency layer audit and repair applied successfully.")
  except Exception as e:
    print(f"Warning: Failed to apply consistency layer audit and repair: {e}")

  # Base64 encode image assets to embed them directly in visualize.html for portability
  import base64
  print("Embedding base64 data for image assets...")
  for asset in spec_data.get("assets", []):
    asset_path = scenario_path / "assets" / asset.get("path", "")
    if not asset_path.exists():
      # Fallback to direct relative path lookup
      asset_path = scenario_path / asset.get("path", "")
    
    if asset_path.is_file():
      try:
        mime_type = asset.get("mime_type") or "image/png"
        with open(asset_path, "rb") as img_f:
          b64_data = base64.b64encode(img_f.read()).decode("utf-8")
        asset["base64_data"] = f"data:{mime_type};base64,{b64_data}"
        print(f"  Embedded asset: {asset.get('id')}")
      except Exception as e:
        print(f"  Warning: Failed to base64 encode asset {asset.get('id')}: {e}")

  # Read HTML template
  script_dir = Path(__file__).parent.resolve()
  template_path = script_dir / "templates" / "visualize_template.html"
  if not template_path.exists():
    print(f"Error: Visualizer HTML template not found: {template_path}")
    sys.exit(1)

  with open(template_path, "r", encoding="utf-8") as f:
    template_content = f.read()

  # Embed spec JSON
  spec_str = json.dumps(spec_data, indent=2, ensure_ascii=False)
  output_content = template_content.replace("/* SCENARIO_DATA_PLACEHOLDER */", spec_str)

  # Save visualize.html
  output_file = scenario_path / "visualize.html"
  with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_content)

  print("\n======================================================================")
  print("SUCCESS: Standalone Scenario Visualizer Compiled!")
  print(f"Open visualizer in browser: file://{output_file}")
  print("======================================================================\n")

if __name__ == "__main__":
  main()
