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

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ChartPreset:
    id: str
    chart_type: str
    title: str
    subtitle: str
    series_names: list[str] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    default_styling: dict[str, Any] = field(default_factory=dict)
    guidance: str = ""
    caption_template_id: str = "operational_scene_update"

@dataclass
class CaptionTemplate:
    id: str
    template_string: str  # Baseline template structure
    required_checklist: list[str] = field(default_factory=list)  # Mandatory semantic elements

CHART_PRESETS = {
    "preset_bar_chart": ChartPreset(
        id="preset_bar_chart",
        chart_type="bar_chart",
        title="[Component Name] Single Metric Status",
        subtitle="Current status or volume count for a single system category",
        y_label="Metric Volume",
        default_styling={"theme": "cool", "show_grid": True},
        guidance="Use when representing single-variable discrete volumes (e.g., active requests per handler, total error count, memory utilization). Each row expects a single 'value'.",
    ),
    "preset_line_chart": ChartPreset(
        id="preset_line_chart",
        chart_type="line_chart",
        title="[Metric Name] Temporal Trend",
        subtitle="Continuous timeline tracking for a single metric trend",
        y_label="Value",
        default_styling={"theme": "cool", "show_grid": True},
        guidance="Use when representing continuous time-series history of a single variable (e.g., latency milliseconds over a 10-minute window). Each row expects a single 'value'.",
    ),
    "preset_multi_line_chart": ChartPreset(
        id="preset_multi_line_chart",
        chart_type="multi_line_chart",
        title="[System Name] Traffic Comparison",
        subtitle="Temporal comparative breakdown between multiple traffic streams",
        series_names=["External Ingress", "Internal Healthchecks"],
        y_label="Throughput / Volume",
        default_styling={"theme": "cool", "show_grid": True},
        guidance="Use when comparing multiple continuous time-series metrics side-by-side (e.g., ingress requests vs health check status, query volume vs cache hits). Each row expects list of 'values' mapping to 'series_names'.",
    ),
    "preset_grouped_bar_chart": ChartPreset(
        id="preset_grouped_bar_chart",
        chart_type="grouped_bar_chart",
        title="[System Category] Resource Split",
        subtitle="Side-by-side comparative allocation showing multiple sub-metrics",
        series_names=["Category A", "Category B"],
        y_label="Count / Amount",
        default_styling={"theme": "warm", "show_grid": True},
        guidance="Use when comparing multiple discrete, side-by-side categories under several labels (e.g., connection pool Active vs Idle, VM CPU vs Memory). Each row expects list of 'values' mapping to 'series_names'.",
    ),
    "preset_stacked_bar_chart": ChartPreset(
        id="preset_stacked_bar_chart",
        chart_type="stacked_bar_chart",
        title="[System Name] Resource Breakdown",
        subtitle="Stacked category breakdown showing proportions contributing to total volume",
        series_names=["Component X", "Component Y"],
        y_label="Total Volume",
        default_styling={"theme": "warm", "show_grid": True},
        guidance="Use when representing stacked categories summing up to a total volume across intervals (e.g., request load breakdown by API endpoint, thread state distribution). Each row expects list of 'values' mapping to 'series_names'.",
    ),
    "preset_pie_chart": ChartPreset(
        id="preset_pie_chart",
        chart_type="pie_chart",
        title="[Component Name] Ratios",
        subtitle="Percentage distribution of statuses or categories",
        default_styling={"show_legend": True},
        guidance="Use to represent proportional share distributions (e.g., HTTP response status codes 2xx/3xx/4xx/5xx percentages, storage usage slice shares). Each row expects a single 'value'.",
    ),
    "preset_scatter_plot": ChartPreset(
        id="preset_scatter_plot",
        chart_type="scatter_plot",
        title="[Metric X] vs [Metric Y] Distribution",
        subtitle="Correlation distribution scatter plot",
        x_label="Variable X",
        y_label="Variable Y",
        default_styling={"theme": "cool"},
        guidance="Use to plot discrete coordinate samples to identify distribution clusters or correlations (e.g. latency vs payload size, CPU vs thread count). Each row expects 'x' and 'y' float coordinates.",
    ),
    "preset_gantt_chart": ChartPreset(
        id="preset_gantt_chart",
        chart_type="gantt_chart",
        title="[Process Name] Step Execution Timeline",
        subtitle="Mitigation or execution step-by-step progress timelines",
        y_label="Execution Step",
        default_styling={"theme": "cool", "show_grid": True},
        guidance="Use to represent concurrent execution phases or sequential tasks over time (e.g., automation script execution steps, service startup initialization phases). Each row expects a 'start' timestamp (integer offset) and 'duration'.",
    ),
}

CAPTION_TEMPLATES = {
    "operational_scene_update": CaptionTemplate(
        id="operational_scene_update",
        # Kept modality-neutral on purpose: this caption is shared by charts,
        # physical natural_image photos, and schematic diagrams, so it must NOT
        # presuppose a "status board" / "the display shows" screen framing.
        template_string="Operational evidence for {service_name}: {visual_anomaly_details}. Visual anchors {anchors} confirm the current state.",
        required_checklist=[
            "Describe what the evidence shows (physical equipment state, chart trend, or diagram element).",
            "Detail the specific visual anomaly (e.g., traffic drop, connection spike, fault indicator).",
            "Explicitly reference all associated anchor IDs.",
        ],
    ),
}
