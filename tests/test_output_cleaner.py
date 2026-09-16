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

"""Tests for online trajectory output cleaner."""

from __future__ import annotations

from clasharness.utils.output_cleaner import clean_openclaw_output


def test_cleaner_removes_compatibility_noise_lines():
    noisy_stdout = "\n".join(
        [
            "◇  Compatibility config keys detected",
            "Run \"openclaw doctor --fix\" to apply compatibility migrations.",
            "Actual useful output",
        ]
    )
    noisy_stderr = "\n".join(
        [
            "Invalid config at /tmp/openclaw/openclaw.json:",
            "error: unknown option '--name'",
        ]
    )
    clean_stdout, clean_stderr = clean_openclaw_output(noisy_stdout, noisy_stderr)
    assert "Compatibility config keys detected" not in clean_stdout
    assert "doctor --fix" not in clean_stdout
    assert "Actual useful output" in clean_stdout
    assert "Invalid config at" not in clean_stderr
    assert "unknown option '--name'" in clean_stderr
