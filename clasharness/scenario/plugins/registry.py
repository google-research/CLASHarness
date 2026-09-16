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

"""Decorator-based registration for conflict plugins.

Mirrors the ``@register_skill`` extension layer in ``clasharness/skills``: a
plugin class decorated with :func:`register_conflict_plugin` is discovered and
instantiated by ``clasharness.scenario.plugins`` without a hand-maintained
list. This module deliberately imports *no* concrete plugin, so plugin modules
can import the decorator from here without a circular import.
"""

from __future__ import annotations

from clasharness.scenario.plugins.base import BaseConflictPlugin

# Registered plugin classes, in decoration (import) order.
_REGISTERED: list[type[BaseConflictPlugin]] = []


def register_conflict_plugin(cls: type[BaseConflictPlugin]) -> type[BaseConflictPlugin]:
    """Class decorator that registers a conflict plugin.

    Registration is idempotent: re-importing a module will not double-register
    the same class.
    """
    if cls not in _REGISTERED:
        _REGISTERED.append(cls)
    return cls


def registered_plugin_classes() -> list[type[BaseConflictPlugin]]:
    """Return the registered plugin classes in import order."""
    return list(_REGISTERED)


def build_registered_plugins() -> list[BaseConflictPlugin]:
    """Instantiate every registered plugin class (import order preserved)."""
    return [cls() for cls in _REGISTERED]
