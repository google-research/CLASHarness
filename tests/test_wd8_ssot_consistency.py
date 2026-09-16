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

"""WD8 -- single-source-of-truth (SSoT) consistency guards for the visual vocab.

WD1-WD6 landed the design-archetype library as the *one* place the visual style
vocabulary, coherent token presets, and per-style render guidance live. These
tests lock that invariant across the modules that consume it, so a future change
cannot quietly reintroduce a *second* vocabulary source (the historical failure
mode: a hardcoded ``suggested_visual_styles`` list in the materializer and a
crude 6-item ``STYLE_PRESETS`` map that drifted from the gallery/contract).

Reproducibility of the sampler itself (same seed -> same design) is covered by
``test_design_archetypes.py`` (UT-ARCH-12/13/16) and the materializer-level
seeding by ``test_vlm_presets_overrides.py`` (UT-WD5-*); this file is about
*cross-module single-sourcing*.
"""

from __future__ import annotations

from pathlib import Path

from clasharness.scenario import conflict_vlm_materializer
from clasharness.scenario import design_archetypes as da
from clasharness.scenario import llm_client
from clasharness.scenario import vlm_materializer


def test_wd8_style_vocabulary_has_a_single_source() -> None:
  # UT-WD8-01: the canonical vocabulary is derived from the archetype library and
  # nowhere else -- the public tuple, the accessor, and the objects agree exactly.
  names_from_objects = tuple(a.name for a in da.all_archetypes())
  assert da.VISUAL_STYLE_NAMES == names_from_objects
  assert tuple(da.archetype_names()) == names_from_objects
  # A merged/parallel source would show up as duplicate names.
  assert len(set(names_from_objects)) == len(names_from_objects)
  # Every advertised name resolves back to itself (no aliases/ghost entries).
  for name in names_from_objects:
    assert da.has_archetype(name)
    assert da.get_archetype(name).name == name


def test_wd8_retired_parallel_sources_stay_retired() -> None:
  # UT-WD8-02: the historical second sources must not come back. Neither
  # materializer may define its own ``STYLE_PRESETS`` map or a hardcoded
  # ``suggested_visual_styles`` vocabulary, and both must import the SSoT module.
  for module in (vlm_materializer, conflict_vlm_materializer, llm_client):
    assert not hasattr(module, "STYLE_PRESETS"), module.__name__
    assert not hasattr(module, "suggested_visual_styles"), module.__name__
    assert getattr(module, "design_archetypes", None) is da, module.__name__


def test_wd8_design_guidance_is_ssot_sourced_and_populated() -> None:
  # UT-WD8-03: the LLM compose path injects design_guidance built solely from
  # ``get_archetype(visual_style)``. Lock that every canonical style yields a
  # populated, non-stub guidance bundle, so no style feeds the renderer an empty
  # or hand-copied design language.
  for name in da.VISUAL_STYLE_NAMES:
    arch = da.get_archetype(name)
    guidance = {
        "visual_style": arch.name,
        "aesthetic_descriptor": arch.aesthetic_descriptor,
        "render_guidance": arch.render_guidance,
    }
    assert guidance["visual_style"] == name
    assert guidance["aesthetic_descriptor"].strip(), name
    # render_guidance is meant to be a detailed paragraph, not a placeholder.
    assert len(guidance["render_guidance"].strip()) > 40, name


def _module_source(module) -> str:
  return Path(module.__file__).read_text(encoding="utf-8")


def test_wd8_llm_client_enum_is_generated_from_ssot() -> None:
  # UT-WD8-04: the compose contract's visual_style enum must be generated from
  # ``design_archetypes.VISUAL_STYLE_NAMES`` rather than a literal list baked into
  # the prompt (which would silently drift as the library grows).
  source = _module_source(llm_client)
  assert "design_archetypes.VISUAL_STYLE_NAMES" in source


def test_wd8_style_gallery_enumerates_the_ssot() -> None:
  # UT-WD8-05: the gallery renders whatever the library contains -- it iterates
  # ``design_archetypes.all_archetypes()`` instead of a hand-maintained copy, so
  # a new archetype automatically gets a gallery tile.
  # No `.resolve()`: under the build system `__file__` is a symlink into the
  # CAS, and resolving it walks out of the runfiles tree.
  gallery = Path(__file__).parents[1] / "scripts" / "generate_style_gallery.py"
  assert gallery.exists(), gallery
  source = gallery.read_text(encoding="utf-8")
  assert "design_archetypes.all_archetypes()" in source
