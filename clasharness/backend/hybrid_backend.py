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

"""HybridBackend — real openclaw gateway + mocked app services.

Extends MultiAppBackend(real_openclaw=True) with gateway lifecycle management:
- Optional auto-start/stop of the openclaw gateway process
- Automatic free-port selection to avoid conflicts
- Readiness polling: waits until `openclaw health` returns 0
- Port injection into subprocess env via OPENCLAW_PORT

Usage:
    # No gateway management (same as MultiAppBackend(real_openclaw=True))
    backend = HybridBackend()

    # Auto-start the gateway on initialize(), stop on cleanup()
    backend = HybridBackend(auto_start_gateway=True, gateway_port=19000)
"""

from __future__ import annotations

import socket
import subprocess
import time
from typing import Any

from clasharness.backend.multi_app_backend import MultiAppBackend


class HybridBackend(MultiAppBackend):
    """Real openclaw CLI + mocked app backends, with gateway lifecycle management.

    Command routing (inherited from MultiAppBackend):
        openclaw * → RealOpenClawBackend (subprocess)
        calendar *  → CalendarBackend (mock)
        gcalcli *   → CalendarBackend (mock)
        email *     → EmailBackend (mock)
        weather *   → WeatherBackend (mock)
        file *      → FileSystemBackend (mock)
        tasks *     → TasksBackend (mock)

    Because this class inherits MultiAppBackend, all evaluation state
    accessors (.calendar_backend, .email_backend, etc.) and
    isinstance(backend, MultiAppBackend) checks work transparently.
    """

    def __init__(
        self,
        auto_start_gateway: bool = False,
        gateway_port: int | None = None,
        gateway_ready_timeout: int = 15,
        skip_incompatible_openclaw: bool = True,
        fallback_openclaw_network_to_mock: bool = False,
        strict_online_data: bool = True,
        rewrite_to_openclaw: bool = False,
    ) -> None:
        """
        Args:
            auto_start_gateway: If True, call `openclaw gateway start` on
                initialize() and terminate the process on cleanup().
            gateway_port: Port for the gateway. Defaults to a randomly
                chosen free port.
            gateway_ready_timeout: Seconds to wait for gateway readiness
                before giving up (does not raise; just logs and continues).
        """
        super().__init__(
            real_openclaw=True,
            real_openclaw_kwargs={
                "skip_incompatible_openclaw": skip_incompatible_openclaw,
                "rewrite_to_openclaw": rewrite_to_openclaw,
            },
            fallback_openclaw_network_to_mock=fallback_openclaw_network_to_mock,
            strict_online_data=strict_online_data,
        )
        self._auto_start = auto_start_gateway
        self._gateway_port: int = gateway_port if gateway_port is not None else _find_free_port()
        self._ready_timeout = gateway_ready_timeout
        self._gateway_proc: subprocess.Popen | None = None
        self._rewrite_to_openclaw = rewrite_to_openclaw

    # ------------------------------------------------------------------ #
    # BaseBackend overrides                                                 #
    # ------------------------------------------------------------------ #

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        """Initialize sub-backends with merged env and optional gateway."""
        import os

        merged = {**os.environ, **env_vars, "OPENCLAW_PORT": str(self._gateway_port)}
        # Keep merged env for health checks and managed gateway startup.
        self._env = merged
        super().initialize(state_dir, merged)

        if self._auto_start:
            self._start_gateway()

    def cleanup(self) -> None:
        self._stop_gateway()
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Gateway lifecycle helpers                                             #
    # ------------------------------------------------------------------ #

    def _start_gateway(self) -> None:
        """Start the openclaw gateway process and poll for readiness."""
        try:
            cmd = (
                ["openclaw", "gateway", "start", "--port", str(self._gateway_port)]
                if self._rewrite_to_openclaw
                else ["system", "gateway", "start", "--port", str(self._gateway_port)]
            )
            self._gateway_proc = subprocess.Popen(
                cmd,
                env=self._env if hasattr(self, "_env") else None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # openclaw not installed — proceed without gateway
            self._gateway_proc = None
            return

        deadline = time.monotonic() + self._ready_timeout
        while time.monotonic() < deadline:
            if self._poll_health():
                return
            time.sleep(0.5)
        # Timeout reached — not fatal, evaluation will just use what's available

    def _stop_gateway(self) -> None:
        """Terminate the managed gateway process if one was started."""
        if self._gateway_proc is None:
            return
        try:
            self._gateway_proc.terminate()
            self._gateway_proc.wait(timeout=5)
        except Exception:
            try:
                self._gateway_proc.kill()
            except Exception:
                pass
        finally:
            self._gateway_proc = None

    def _poll_health(self) -> bool:
        """Return True if `openclaw health` exits 0."""
        env = getattr(self, "_env", None)
        try:
            cmd = ["openclaw", "health"] if self._rewrite_to_openclaw else ["system", "health"]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3,
                env=env,
            )
            return proc.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def gateway_port(self) -> int:
        """The port this backend uses (or would use) for the openclaw gateway."""
        return self._gateway_port

    @property
    def gateway_process(self) -> subprocess.Popen | None:
        """The managed gateway subprocess, or None if not started."""
        return self._gateway_proc

    @property
    def is_gateway_managed(self) -> bool:
        """True if auto_start_gateway=True was passed at construction."""
        return self._auto_start

    @property
    def mock_backend(self) -> Any:
        """Satisfies inheritance check in tests."""
        return None

    @property
    def real_backend(self) -> BaseBackend | None:
        """Satisfies inheritance check in tests."""
        return self


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
