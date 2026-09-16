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

"""Mathematical curve generators for creating realistic time-series data."""

import math
import random
import typing


def generate_pattern_data(
    pattern: str,
    params: dict[str, typing.Any],
    length: int,
) -> list[float]:
  """Generate a float list representing a mathematical trend curve."""
  baseline = float(params.get("baseline", 100.0))
  noise_std = float(params.get("noise_std", 0.0))

  values = []
  if pattern in {"flatline", "steady"}:
    for _ in range(length):
      values.append(baseline + random.normalvariate(0, noise_std))

  elif pattern in {"spike", "drop"}:
    spike_val = float(params.get("spike_value", 0.0))
    spike_idx = int(params.get("spike_index", length // 2))
    for i in range(length):
      base = spike_val if i == spike_idx else baseline
      values.append(base + random.normalvariate(0, noise_std))

  elif pattern == "decay":
    start_val = float(params.get("start_value", baseline))
    end_val = float(params.get("end_value", 0.0))
    for i in range(length):
      t = i / (length - 1) if length > 1 else 1.0
      base = start_val + (end_val - start_val) * t
      values.append(base + random.normalvariate(0, noise_std))

  elif pattern in {"periodic", "wave"}:
    amplitude = float(params.get("amplitude", 20.0))
    freq = float(params.get("frequency", 1.0))
    for i in range(length):
      base = baseline + amplitude * math.sin(
          2 * math.pi * freq * (i / length)
      )
      values.append(base + random.normalvariate(0, noise_std))

  else:
    # Fallback to simple baseline
    values = [baseline] * length

  if not params.get("allow_negative", False):
    values = [max(0.0, v) for v in values]

  return values
