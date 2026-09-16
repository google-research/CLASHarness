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

"""Generate an interactive HTML gallery containing every visual UI style.

The style list is sourced from the design-archetype SSoT
(``clasharness.scenario.design_archetypes``) rather than a hand-maintained copy,
so the gallery always matches the canonical archetype library.
"""

import json
import os
import sys
from pathlib import Path

# Resolve absolute paths
ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT_DIR / "clasharness" / "scenario" / "templates" / "chart_template.html"
OUTPUT_DIR = ROOT_DIR / "clasharness" / "data" / "style_gallery"

# Ensure the project root is importable when run as a standalone script.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from clasharness.scenario import design_archetypes  # noqa: E402


def get_mock_spec(style_name, theme, primary_color, style_details):
    series_names = ["Active Connections", "Queue Depth"]
    labels = ["10:45", "10:47", "10:49", "10:51", "10:53", "10:55", "10:57", "10:59"]
    active_data = [25, 40, 85, 100, 100, 100, 100, 100]
    queue_data = [0, 0, 5, 45, 95, 150, 150, 150]
    
    rows = []
    for i, label in enumerate(labels):
        rows.append({
            "label": label,
            "values": [active_data[i], queue_data[i]],
            "Active Connections": active_data[i],
            "Queue Depth": queue_data[i]
        })

    return {
        "title": "Database Connection Pool Status",
        "subtitle": f"Live telemetry node - visual_style: {style_name}",
        "chart_type": "multi_line_chart",
        "series_names": series_names,
        "rows": rows,
        "dashboard_data": {
            "system_name": f"{style_name.replace('_', ' ').title()} Telemetry",
            "cluster_region": "us-central1-a",
            "operator_email": "sre-alerts@ops.example.com",
            "node_hostname": f"node-prod-{style_name}-01.internal",
            "services": [
                {"name": "Auth API Gateway", "status": "healthy", "details": "latency 12ms"},
                {"name": "Session Cache Sync", "status": "healthy", "details": "hit_rate 98%"},
                {"name": "MySQL Core Shard 01", "status": "critical", "details": "pool exhausted"},
                {"name": "Redis Invalidate PubSub", "status": "healthy", "details": "idle"}
            ],
            "metrics": [
                {"name": "Node Latency", "value": "18.4 ms", "sparkline": [10, 15, 25, 12, 18]},
                {"name": "CPU Workload", "value": "78%", "sparkline": [40, 55, 88, 60, 78]},
                {"name": "Memory Buffer", "value": "64.2 GB", "sparkline": [64, 64, 64, 64, 64]},
                {"name": "Active Channels", "value": "249/s", "sparkline": [120, 200, 310, 220, 249]}
            ]
        },
        "layout": "devops_dashboard",
        "custom_overrides": {
            "styling": {
                "theme": theme,
                "primary_color": primary_color,
                "visual_style": style_name,
                "style_details": style_details
            }
        }
    }

def load_spec_from_metadata(metadata_path, style_name, theme, primary_color, style_details):
    import json
    from pathlib import Path
    
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    provider_metadata = meta.get("provider_metadata", {})
    chart_rows = provider_metadata.get("chart_rows", [])
    
    series_names = []
    if chart_rows:
        keys = list(chart_rows[0].keys())
        series_names = [k for k in keys if k not in ("time", "timestamp", "label")]
    else:
        series_names = ["Value"]
        
    rows = []
    for r in chart_rows:
        label = r.get("time") or r.get("timestamp") or r.get("label") or ""
        values = [r.get(s, 0) for s in series_names]
        row_entry = {
            "label": str(label),
            "values": values
        }
        for s in series_names:
            row_entry[s] = r.get(s, 0)
        rows.append(row_entry)
        
    title = meta.get("id", "").replace("_", " ").title()
    if not title:
        title = "Database Connection Pool Status"
        
    return {
        "title": title,
        "subtitle": f"Live telemetry node - visual_style: {style_name}",
        "chart_type": "multi_line_chart" if len(series_names) > 1 else "line_chart",
        "series_names": series_names,
        "rows": rows,
        "dashboard_data": {
            "system_name": f"{style_name.replace('_', ' ').title()} Telemetry",
            "cluster_region": "us-central1-a",
            "operator_email": "sre-alerts@ops.example.com",
            "node_hostname": f"node-prod-{style_name}-01.internal",
            "services": [
                {"name": "Auth API Gateway", "status": "healthy", "details": "latency 12ms"},
                {"name": "Session Cache Sync", "status": "healthy", "details": "hit_rate 98%"},
                {"name": "MySQL Core Shard 01", "status": "critical", "details": "pool exhausted"},
                {"name": "Redis Invalidate PubSub", "status": "healthy", "details": "idle"}
            ],
            "metrics": [
                {"name": "Node Latency", "value": "18.4 ms", "sparkline": [10, 15, 25, 12, 18]},
                {"name": "CPU Workload", "value": "78%", "sparkline": [40, 55, 88, 60, 78]},
                {"name": "Memory Buffer", "value": "64.2 GB", "sparkline": [64, 64, 64, 64, 64]},
                {"name": "Active Channels", "value": "249/s", "sparkline": [120, 200, 310, 220, 249]}
            ]
        },
        "layout": "devops_dashboard",
        "custom_overrides": {
            "styling": {
                "theme": theme,
                "primary_color": primary_color,
                "visual_style": style_name,
                "style_details": style_details
            }
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate style gallery switcher.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Path to scenario asset metadata JSON to load instead of mock data.")
    args = parser.parse_args()

    print("Initializing style gallery output directory...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Base chart template not found at {TEMPLATE_PATH}")

    template_content = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Visual configurations for every archetype, sourced from the SSoT library.
    styles_config = [
        {
            "name": archetype.name,
            "theme": archetype.default_theme,
            "primary_color": archetype.default_primary_color,
            "details": archetype.default_style_details(),
        }
        for archetype in design_archetypes.all_archetypes()
    ]

    for config in styles_config:
        style_name = config["name"]
        print(f"Generating static file for style: {style_name}...")
        if args.metadata_path:
            spec = load_spec_from_metadata(args.metadata_path, style_name, config["theme"], config["primary_color"], config["details"])
        else:
            spec = get_mock_spec(style_name, config["theme"], config["primary_color"], config["details"])
        
        injected_spec = f"window.CHART_SPEC = {json.dumps(spec, ensure_ascii=False)};"
        output_content = template_content.replace("// INJECTED SPEC:", injected_spec)
        
        output_file = OUTPUT_DIR / f"style_{style_name}.html"
        output_file.write_text(output_content, encoding="utf-8")
        print(f"  Saved style file to {output_file}")

    # Build master switcher index.html
    print("Compiling index.html master switcher page...")
    sidebar_links = ""
    for config in styles_config:
        style_name = config["name"]
        display_label = style_name.replace("_", " ").title()
        sidebar_links += f"""
        <button onclick="switchStyle('{style_name}')" id="btn-{style_name}" class="style-btn w-full text-left px-4 py-3 rounded-lg text-sm font-semibold transition text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800">
          {display_label}
        </button>
        """

    default_style = styles_config[0]["name"]

    index_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CLASHarness UI Style Gallery</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }}
  </style>
</head>
<body class="bg-slate-50 dark:bg-slate-900 flex h-full w-full">

  <!-- Sidebar Menu Switcher -->
  <aside class="w-80 bg-white dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800 flex flex-col h-full shrink-0">
    <div class="p-6 border-b border-slate-200 dark:border-slate-800">
      <h1 class="text-lg font-bold text-slate-800 dark:text-slate-100">CLASHarness UI</h1>
      <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">Multi-Era Visual Style Catalog</p>
    </div>
    <nav class="flex-1 overflow-y-auto p-4 space-y-1">
      {sidebar_links}
    </nav>
    <div class="p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
      <div class="text-2xs text-slate-400 dark:text-slate-500 text-center">
        TAG=agy | Dynamic Micro-Styles Integration
      </div>
    </div>
  </aside>

  <!-- Gallery Frame Viewer -->
  <main class="flex-1 h-full flex flex-col bg-slate-100 dark:bg-slate-900 p-8">
    <div class="flex items-center justify-between mb-4 shrink-0">
      <div class="flex items-center space-x-3">
        <span class="text-sm font-semibold text-slate-500 dark:text-slate-400">Current Preview:</span>
        <span id="active-style-label" class="px-3 py-1 bg-indigo-100 text-indigo-800 dark:bg-indigo-900/40 dark:text-indigo-300 text-xs font-bold rounded-full"></span>
      </div>
      <a id="direct-link" href="#" target="_blank" class="text-xs text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-300 font-semibold underline">
        Open standalone style page
      </a>
    </div>
    
    <div class="flex-1 bg-white dark:bg-black rounded-2xl shadow-xl overflow-hidden border border-slate-200 dark:border-slate-800 relative">
      <iframe id="viewer-iframe" src="" class="w-full h-full border-none"></iframe>
    </div>
  </main>

  <script>
    function switchStyle(styleName) {{
      // Update iframe source
      const filename = 'style_' + styleName + '.html';
      document.getElementById('viewer-iframe').src = filename;
      
      // Update header labels
      const displayName = styleName.replace(/_/g, ' ').toUpperCase();
      document.getElementById('active-style-label').innerText = displayName;
      document.getElementById('direct-link').href = filename;

      // Update button highlights
      document.querySelectorAll('.style-btn').forEach(btn => {{
        btn.classList.remove('bg-indigo-50', 'text-indigo-700', 'dark:bg-indigo-950/40', 'dark:text-indigo-400');
        btn.classList.add('text-slate-700', 'hover:bg-slate-100', 'dark:text-slate-300', 'dark:hover:bg-slate-800');
      }});
      
      const activeBtn = document.getElementById('btn-' + styleName);
      if (activeBtn) {{
        activeBtn.classList.remove('text-slate-700', 'hover:bg-slate-100', 'dark:text-slate-300', 'dark:hover:bg-slate-800');
        activeBtn.classList.add('bg-indigo-50', 'text-indigo-700', 'dark:bg-indigo-950/40', 'dark:text-indigo-400');
      }}
    }}

    // Initialize with default style
    switchStyle('{default_style}');
  </script>
</body>
</html>
"""
    index_file = OUTPUT_DIR / "index.html"
    index_file.write_text(index_html, encoding="utf-8")
    print(f"Master switcher index.html written successfully to {index_file}")
    print("\nVisual style gallery generated successfully! Go view the switcher gallery:")
    print(f"  file://{index_file.resolve()}")


if __name__ == "__main__":
    main()
