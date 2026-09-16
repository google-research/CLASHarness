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

"""Task generators package.

Single source of truth for *which* generators exist. Importing this package
(or any submodule of it) auto-discovers and imports every generator module in
this directory, which runs each module's ``@BaseTaskGenerator.register(...)``
decorator as a side effect and populates the registry.

Do NOT maintain hardcoded ``import clasharness.tasks.generators.<mod>`` lists in
scripts or tests to trigger registration -- import this package instead. Adding a
new ``<name>.py`` here is automatically picked up; no edits elsewhere are needed.
"""

from __future__ import annotations

import importlib
import pkgutil


def _import_all_submodules() -> None:
    """Import every non-private submodule so its @register decorators run."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module_info.name}")


# Runs once (module import is cached) the first time anything under this package
# is imported.
_import_all_submodules()
