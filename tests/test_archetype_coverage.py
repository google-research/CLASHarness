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

"""Coverage tests for the 15-archetype conflict taxonomy.

These tests are the enforcement mechanism for the conflict-expansion milestone
goal that every one of the 15 target archetypes ships a seed and that all five
top-level categories from ``docs/task-list.md`` are represented. They assert
that:

* the authoritative ``ARCHETYPE_CATEGORY`` map has exactly the 15 archetype
  tokens, each mapped to one of the five ``CONFLICT_CATEGORIES``,
* every one of those 15 archetype tokens is realized by at least one shipped
  seed (15/15 archetype coverage),
* every shipped seed's non-empty ``archetype`` token is a recognized token (so a
  typo'd or orphaned archetype is caught),
* ``category_coverage`` reports every one of the five categories with a count of
  at least one (5/5 category coverage).

Seeds with an empty ``archetype`` are the original pre-expansion seeds that
predate the taxonomy; they are allowed and are expected to bucket under
``"(other)"``.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import clasharness
from clasharness.scenario.conflict_vlm_spec import ARCHETYPE_CATEGORY
from clasharness.scenario.conflict_vlm_spec import CONFLICT_CATEGORIES
from clasharness.scenario.conflict_vlm_spec import category_coverage
from clasharness.scenario.conflict_vlm_spec import load_conflict_vlm_seed_file

# No ``.resolve()``: under sandboxed build runners ``__file__`` can be a symlink
# into a content-addressed store, so resolving it walks out of the tree that
# actually holds the seed directory.
_SEED_DIR = (
    Path(clasharness.__file__).parent
    / "data"
    / "scenario_seeds"
    / "conflict_vlm_v1"
)

# The taxonomy targets exactly 15 distinct archetypes (docs/conflict-expansion-plan.md).
_EXPECTED_ARCHETYPE_COUNT = 15


def _load_seeds():
  files = sorted(_SEED_DIR.glob("*.json"))
  return [load_conflict_vlm_seed_file(p) for p in files]


class ArchetypeCategoryMapTest(unittest.TestCase):
  """Invariants of the authoritative token -> category map itself."""

  def test_map_has_exactly_fifteen_archetypes(self):
    self.assertEqual(
        len(ARCHETYPE_CATEGORY),
        _EXPECTED_ARCHETYPE_COUNT,
        f"expected {_EXPECTED_ARCHETYPE_COUNT} archetype tokens, "
        f"got {sorted(ARCHETYPE_CATEGORY)}",
    )

  def test_every_mapped_category_is_canonical(self):
    unknown = {
        token: cat
        for token, cat in ARCHETYPE_CATEGORY.items()
        if cat not in CONFLICT_CATEGORIES
    }
    self.assertEqual(unknown, {}, f"tokens mapped to unknown categories: {unknown}")

  def test_every_category_has_at_least_one_archetype(self):
    mapped = set(ARCHETYPE_CATEGORY.values())
    missing = [cat for cat in CONFLICT_CATEGORIES if cat not in mapped]
    self.assertEqual(missing, [], f"categories with no archetype in the map: {missing}")


class SeedArchetypeCoverageTest(unittest.TestCase):
  """The shipped seed set realizes the full taxonomy."""

  def setUp(self):
    self.seeds = _load_seeds()
    self.assertTrue(self.seeds, f"no seeds discovered under {_SEED_DIR}")

  def test_every_archetype_token_has_a_seed(self):
    present = {
        (s.archetype or "").strip() for s in self.seeds if (s.archetype or "").strip()
    }
    missing = [token for token in ARCHETYPE_CATEGORY if token not in present]
    self.assertEqual(
        missing, [], f"archetype tokens with no shipped seed: {missing}"
    )

  def test_no_seed_has_an_unrecognized_archetype(self):
    # An empty archetype is allowed (original pre-expansion seeds); a non-empty
    # token that is not in the authoritative map is a typo/orphan and fails.
    orphans = {
        s.slug: s.archetype
        for s in self.seeds
        if (s.archetype or "").strip()
        and (s.archetype or "").strip() not in ARCHETYPE_CATEGORY
    }
    self.assertEqual(orphans, {}, f"seeds with unrecognized archetype tokens: {orphans}")

  def test_category_coverage_reports_all_five_categories(self):
    coverage = category_coverage(self.seeds)
    for category in CONFLICT_CATEGORIES:
      self.assertGreaterEqual(
          coverage.get(category, 0),
          1,
          f"category {category!r} has no seed; coverage={coverage}",
      )

  def test_category_coverage_total_matches_seed_count(self):
    coverage = category_coverage(self.seeds)
    self.assertEqual(
        sum(coverage.values()),
        len(self.seeds),
        f"coverage buckets {coverage} do not sum to {len(self.seeds)} seeds",
    )


if __name__ == "__main__":
  unittest.main()
