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

"""Asset lookup for the eval visualizer, in eval_report and the standalone script.

Both builders look for scenario packages under ``clasharness/data/scenarios``.
A scenario that is not there -- the normal case for downloaded task data, and
for a fresh checkout, which ships no scenarios at all -- must yield no image
references, not an exception: the lookup used to fall back to an undefined
name, and the NameError aborted the whole dashboard.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from train_eval import eval_report  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "_eval_visualizer_under_test", _REPO / "scripts" / "generate_eval_visualizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILDERS = {
    "eval_report": eval_report._load_asset_refs,
    "script": _load_script()._load_asset_paths,
}


def _run(slug: str) -> dict:
    return {"task_id": f"conflict_vlm_{slug}_r01",
            "metadata": {"generator_id": "conflict_vlm_v1", "scenario_slug": slug}}


def _write_scenario(root: Path, slug: str) -> Path:
    pkg = root / "clasharness" / "data" / "scenarios" / "conflict_vlm_v1" / slug
    (pkg / "assets" / "charts").mkdir(parents=True)
    (pkg / "assets" / "charts" / "latency.png").write_bytes(b"\x89PNG fake")
    manifest = {"assets": [
        {"id": "a1", "path": "charts/latency.png"},
        {"id": "a2", "path": "charts/missing.png"},  # listed, not on disk
    ]}
    (pkg / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pkg


@pytest.mark.parametrize("builder", sorted(_BUILDERS))
def test_missing_scenario_yields_no_refs(builder, tmp_path, monkeypatch):
    monkeypatch.setenv("CLASHARNESS_REPO_ROOT", str(tmp_path / "empty_root"))
    assert _BUILDERS[builder]([_run("no_such_scenario_anywhere")], tmp_path) == {}


@pytest.mark.parametrize("builder", sorted(_BUILDERS))
def test_scenario_under_repo_root_yields_relative_refs(builder, tmp_path, monkeypatch):
    root = tmp_path / "root"
    _write_scenario(root, "sla_support_response")
    monkeypatch.setenv("CLASHARNESS_REPO_ROOT", str(root))
    out_dir = tmp_path / "eval_out"
    out_dir.mkdir()

    refs = _BUILDERS[builder]([_run("sla_support_response"), _run("no_such_scenario_anywhere")], out_dir)

    assert set(refs) == {"sla_support_response"}
    entry = refs["sla_support_response"]
    expected = os.path.relpath(
        root / "clasharness/data/scenarios/conflict_vlm_v1/sla_support_response/assets/charts/latency.png",
        out_dir,
    )
    assert entry["blobs"] == {"a1": Path(expected).as_posix()}
    assert entry["alias"] == {"charts/latency.png": "a1", "latency.png": "a1"}


def test_dashboard_is_written_when_no_scenario_is_found(tmp_path, monkeypatch, capsys):
    # The regression as it showed up in eval logs: "Warning: Failed to
    # auto-generate visualizer: name 'py_file_path' is not defined".
    monkeypatch.setenv("CLASHARNESS_REPO_ROOT", str(tmp_path / "empty_root"))
    run = dict(_run("no_such_scenario_anywhere"), success=False, score=0.0)
    (tmp_path / "results.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")

    eval_report._generate_evaluation_visualizer(tmp_path)

    out = capsys.readouterr().out
    assert "Failed to auto-generate visualizer" not in out
    html = (tmp_path / "visualize.html").read_text(encoding="utf-8")
    assert "no_such_scenario_anywhere" in html
