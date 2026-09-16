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

"""Drop-in package for user-contributed extension skills.

Any module imported here that decorates a skill class with
:func:`clasharness.skills.extensions.register_skill` is auto-loaded by
:class:`~clasharness.backend.multi_app_backend.MultiAppBackend` at construction
time and registered alongside the built-in skills.

This package ships empty by default (the core arena has no extensions), so
importing it is a harmless no-op. To add a skill, place your module in this
directory and import it below so the decorator runs::

    from clasharness.skills.contrib import my_skill  # noqa: F401

See ``docs/mcp-extension-guide.md`` for the full guide and a copy-paste
template.
"""
