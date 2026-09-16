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

"""Shared pytest setup for the CLASHarness test suite.

Currently one job: keep provider configuration from leaking between test files.
"""
from __future__ import annotations

import os

import pytest

# Environment variables that PRODUCTION code writes directly -- ``sync_judge_env``
# and the CLI's provider wiring use ``os.environ.setdefault`` / assignment rather
# than a monkeypatched setter, so pytest's own monkeypatch teardown does not undo
# them. Left alone, a test file that configures a judge (say
# ``LLM_JUDGE_PROVIDER=gemini`` with no key) silently reconfigures every later
# file in the same process, which made ``pytest tests/test_llm_judge.py
# tests/test_train_and_eval_cli.py`` fail 40 tests that both files pass alone.
_PROVIDER_ENV = (
    "LLM_JUDGE_PROVIDER",
    "LLM_JUDGE_MODEL",
    "LLM_JUDGE_REQUEST_RETRIES",
    "LLM_JUDGE_RETRY_BACKOFF_S",
    "VERTEX_PROJECT",
    "VERTEX_LOCATION",
    "OPENAI_BASE_URL",
    "BEDROCK_OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _restore_provider_env():
    """Snapshot and restore the provider knobs around every test.

    Scoped to this specific list rather than the whole environment so that
    module- and session-scoped fixtures, which legitimately set other variables
    during the first test that uses them, are not torn down underneath the
    tests that follow.
    """
    saved = {name: os.environ.get(name) for name in _PROVIDER_ENV}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
