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

"""Unit tests for the ``CLASHARNESS_*`` environment-knob readers.

These knobs are set from outside this repository, so a misspelling fails
silently: an unset variable falls back to a default rather than raising. These
tests pin the read semantics -- in particular that set-but-empty is not the
same as unset, and that an unparsable integer degrades to the default instead
of aborting a long run.
"""

import pytest

from clasharness.env_knobs import PREFIX, getenv, getenv_int

_NAME = "CLASHARNESS_TEST_KNOB"


@pytest.fixture(autouse=True)
def _clear_knobs(monkeypatch):
    """Start every test from a known-unset state."""
    monkeypatch.delenv(_NAME, raising=False)


def test_prefix_is_the_project_prefix():
    """Call sites spell knobs with this prefix; a drift here is a rename bug."""
    assert PREFIX == "CLASHARNESS_"
    assert _NAME.startswith(PREFIX)


# --------------------------------------------------------------------------- #
# getenv                                                                       #
# --------------------------------------------------------------------------- #


def test_getenv_reads_the_variable(monkeypatch):
    monkeypatch.setenv(_NAME, "value")
    assert getenv(_NAME) == "value"


def test_getenv_returns_default_when_unset():
    assert getenv(_NAME, "fallback") == "fallback"
    assert getenv(_NAME) is None


def test_getenv_treats_empty_value_as_set(monkeypatch):
    """Matches ``os.environ.get``: set-but-empty is not the same as unset."""
    monkeypatch.setenv(_NAME, "")
    assert getenv(_NAME, "fallback") == ""


# --------------------------------------------------------------------------- #
# getenv_int                                                                   #
# --------------------------------------------------------------------------- #


def test_getenv_int_parses_the_variable(monkeypatch):
    monkeypatch.setenv(_NAME, "17")
    assert getenv_int(_NAME, 3) == 17


def test_getenv_int_returns_default_when_unset():
    assert getenv_int(_NAME, 3) == 3


@pytest.mark.parametrize("raw", ["", "abc", "1.5", " "])
def test_getenv_int_treats_unparsable_values_as_absent(monkeypatch, raw):
    """A typo in an operational knob must not abort a long generation run."""
    monkeypatch.setenv(_NAME, raw)
    assert getenv_int(_NAME, 3) == 3
