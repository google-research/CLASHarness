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

"""Layer-2b opt-in *live* tests: real-mode commands really executing on Linux.

Phase 3 context
---------------
Layers 1 and 2a cover every real-path branch hermetically (subprocess skills, and
the pure helpers of the external skills). This module closes the loop for the
skills whose real path is already wired and can genuinely run on a plain Linux
host *without credentials*: it drives them through the real-mode backend exactly
as the MCP layer does (``backend.execute_cli(...)``) and asserts they actually hit
the network / shell out — not the mock.

Scope
-----
  * ``curl``   -> real ``HttpBackend`` (``_real_http=True`` in ``mode="real"``)
    shells out to the real ``curl`` binary against a public endpoint.
  * ``weather``-> its online path shells out to ``curl`` against the free
    open-meteo API (no API key), so ``weather get --online`` returns live data.

calendar / email / tasks are intentionally excluded: their real providers
(gcalcli / khal / Google API / himalaya) require credentials or config that do
not exist by default, so they cannot "really execute" here. They stay covered by
Layer 2a (pure helpers) plus their mock tests.

Opt-in and no-op by default
---------------------------
Every test calls ``pytest.skip()`` at runtime unless ``RUN_LIVE_EXTERNAL_TESTS``
is set (e.g. ``RUN_LIVE_EXTERNAL_TESTS=1``). Runtime skips (not
``@pytest.mark.skipif``) are used so the suite is a no-op under both the local
bare-``unittest`` runner and pytest. When opted in on a host with no
network egress, the tests skip (rather than fail) with an explicit reason, so a
green run always means the real path genuinely executed.
"""

from __future__ import annotations

import os
import shutil

import pytest

from clasharness.core.environment import make_env

# A task with workspace support (shared with the other Phase 3 suites).
_TASK_ID = "calendar_add_event_1"

_LIVE_ENV_FLAG = "RUN_LIVE_EXTERNAL_TESTS"
_LIVE_TRUTHY = {"1", "true", "yes", "on"}


def _require_live() -> None:
    """Skip unless the operator explicitly opted into live network tests."""
    if os.environ.get(_LIVE_ENV_FLAG, "").strip().lower() not in _LIVE_TRUTHY:
        pytest.skip(f"live external tests disabled (set {_LIVE_ENV_FLAG}=1 to enable)")


def _require_curl() -> None:
    if shutil.which("curl") is None:
        pytest.skip("curl binary not available on host")


def _real_backend():
    env = make_env(_TASK_ID, mode="real")
    env.reset()
    return env._backend


# --------------------------------------------------------------------------- #
# curl: raw real HTTP through the real-mode backend                             #
# --------------------------------------------------------------------------- #
def test_live_curl_real_http_hits_public_api():
    _require_live()
    _require_curl()
    be = _real_backend()

    # Sanity: real mode must have selected the real curl path, not the mock one.
    assert be._http._real_http is True

    res = be.execute_cli(
        'curl -sG "https://api.open-meteo.com/v1/forecast" '
        '--data-urlencode "latitude=51.5074" '
        '--data-urlencode "longitude=-0.1278" '
        '--data-urlencode "current=temperature_2m"'
    )

    # The command must have gone through the *real* HTTP path regardless of egress.
    assert res.meta is not None
    assert res.meta.get("http_mode") == "real"

    if res.exit_code != 0:
        pytest.skip(f"no network egress for real curl: {res.stderr.strip() or res.exit_code}")

    # With egress, the live open-meteo API returns JSON with a 'current' block.
    assert '"current"' in res.stdout
    assert '"temperature_2m"' in res.stdout


# --------------------------------------------------------------------------- #
# weather: real online fetch through the real-mode backend                      #
# --------------------------------------------------------------------------- #
def test_live_weather_real_online_fetches_open_meteo():
    _require_live()
    _require_curl()
    be = _real_backend()

    # Real mode enables online data (+ strict), so this shells out to open-meteo.
    assert be._weather_skill._enable_online_data is True

    res = be.execute_cli("weather get --location London --online")

    if res.exit_code != 0:
        # Strict online mode disables mock fallback, so a network failure surfaces
        # as an error rather than a mock result. Treat that as "no egress" -> skip.
        pytest.skip(f"open-meteo unreachable in real mode: {res.stderr.strip() or res.exit_code}")

    # A zero exit under strict online mode can only come from a live fetch.
    assert "Source    : online" in res.stdout
    assert "Condition" in res.stdout


if __name__ == "__main__":  # pragma: no cover - convenience for direct runs
    import sys

    sys.exit(0)
