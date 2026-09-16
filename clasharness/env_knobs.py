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

"""Reads for the ``CLASHARNESS_*`` operational tuning knobs.

These knobs are set from outside the repository -- shell profiles, launcher
scripts, job configs -- so a misspelling fails *silently*: an unset variable
falls back to a default instead of raising, and a run keeps going with, say,
the default worker count and nothing in the logs to say why. Reading them
through one place at least keeps the spellings in a single grep.

This module deliberately imports nothing from ``clasharness`` so that it stays
safe to import from anywhere in the package, including modules that run while
``clasharness/__init__`` is still executing.
"""

from __future__ import annotations

import os

PREFIX = "CLASHARNESS_"


def getenv(name: str, default: str | None = None) -> str | None:
    """``os.environ.get`` for a ``CLASHARNESS_*`` knob.

    A variable explicitly set to the empty string counts as set and is *not*
    replaced by ``default``, matching ``os.environ.get`` semantics.
    """
    value = os.environ.get(name)
    return default if value is None else value


def getenv_int(name: str, default: int) -> int:
    """:func:`getenv` parsed as an ``int``, falling back to ``default``.

    An unparsable value is treated as absent rather than fatal: these knobs are
    operational tuning, and a typo should not take down a long run.
    """
    raw = getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
