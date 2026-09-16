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

"""WD6 unit tests for the render quality floor (blank/collapsed detection).

These tests exercise the pure measurement helper ``_render_quality_report`` and
the enforcing wrapper ``_assert_render_quality`` in ``vlm_asset_generators``.
They build synthetic PNGs with PIL, so they need neither Chrome nor a network
and run fast/deterministically.
"""

import io
import os

from PIL import Image, ImageDraw

from clasharness.scenario.vlm_asset_generators import _assert_render_quality
from clasharness.scenario.vlm_asset_generators import _render_quality_report
from clasharness.scenario.vlm_asset_generators import RenderQualityError

_ESCAPE_HATCH = "CLASHARNESS_DISABLE_RENDER_QUALITY_CHECK"


def _png_bytes(image: Image.Image) -> bytes:
  buf = io.BytesIO()
  image.save(buf, format="PNG")
  return buf.getvalue()


def _rich_page(size: tuple[int, int] = (400, 300)) -> bytes:
  """A believable UI-ish page: white canvas with many colored blocks + text."""
  image = Image.new("RGB", size, "white")
  draw = ImageDraw.Draw(image)
  palette = [
      (30, 64, 175),
      (220, 38, 38),
      (22, 163, 74),
      (234, 179, 8),
      (147, 51, 234),
      (14, 165, 233),
  ]
  for idx, color in enumerate(palette):
    x0 = 10 + (idx % 3) * 130
    y0 = 10 + (idx // 3) * 130
    draw.rectangle([x0, y0, x0 + 110, y0 + 110], fill=color)
  draw.text((20, 270), "Dashboard :: revenue / latency / errors", fill="black")
  return _png_bytes(image)


# ---------------------------------------------------------------------------
# _render_quality_report (pure measurement)
# ---------------------------------------------------------------------------


def test_rq01_blank_white_page_is_rejected():
  """UT-RQ-01: a flat white page has no ink and must fail the floor."""
  report = _render_quality_report(_png_bytes(Image.new("RGB", (400, 300), "white")))
  assert report.ok is False
  assert "blank" in report.reason


def test_rq02_single_flat_color_is_rejected():
  """UT-RQ-02: a single non-white flat color is still a blank render."""
  solid = Image.new("RGB", (400, 300), (30, 64, 175))
  report = _render_quality_report(_png_bytes(solid))
  assert report.ok is False
  assert report.distinct_colors <= 1
  assert "blank" in report.reason


def test_rq03_degenerate_dimensions_are_rejected():
  """UT-RQ-03: a collapsed screenshot below the min dimension fails on size."""
  tiny = Image.new("RGB", (8, 8), "white")
  ImageDraw.Draw(tiny).rectangle([1, 1, 6, 6], fill=(220, 38, 38))
  report = _render_quality_report(_png_bytes(tiny))
  assert report.ok is False
  assert "degenerate dimensions" in report.reason


def test_rq04_undecodable_bytes_are_rejected():
  """UT-RQ-04: garbage that is not a decodable image is a hard failure."""
  report = _render_quality_report(b"this is definitely not a PNG")
  assert report.ok is False
  assert "undecodable" in report.reason


def test_rq05_fully_transparent_page_reads_as_blank():
  """UT-RQ-05: transparent pixels composite over white, so an empty RGBA page
  is blank rather than a single 'transparent' color."""
  transparent = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
  report = _render_quality_report(_png_bytes(transparent))
  assert report.ok is False
  assert "blank" in report.reason


def test_rq06_rich_real_content_passes():
  """UT-RQ-06: a page with several colors and real ink clears the floor."""
  report = _render_quality_report(_rich_page())
  assert report.ok is True
  assert report.non_background_fraction >= 0.002
  assert report.width == 400 and report.height == 300


def test_rq07_small_but_sufficient_ink_passes():
  """UT-RQ-07: even a minimalist page (one modest block) clears the floor."""
  image = Image.new("RGB", (400, 300), "white")
  # ~60x60 block on a 400x300 canvas is ~3% of area -- well above the 0.2% floor.
  ImageDraw.Draw(image).rectangle([20, 20, 80, 80], fill=(30, 64, 175))
  report = _render_quality_report(_png_bytes(image))
  assert report.ok is True
  assert report.non_background_fraction >= 0.002


def test_rq08_report_is_deterministic():
  """UT-RQ-08: measurement is pure -- identical bytes give identical reports."""
  data = _rich_page()
  first = _render_quality_report(data)
  second = _render_quality_report(data)
  assert first == second


# ---------------------------------------------------------------------------
# _assert_render_quality (enforcement + escape hatch)
# ---------------------------------------------------------------------------


def test_rq09_assert_raises_with_context_on_blank():
  """UT-RQ-09: the gate raises RenderQualityError tagged with its context."""
  blank = _png_bytes(Image.new("RGB", (400, 300), "white"))
  try:
    _assert_render_quality(blank, context="raw_llm_html render")
  except RenderQualityError as exc:
    assert "raw_llm_html render" in str(exc)
  else:
    raise AssertionError("expected RenderQualityError for a blank render")


def test_rq10_assert_passes_for_rich_content():
  """UT-RQ-10: a healthy render passes the gate without raising."""
  _assert_render_quality(_rich_page(), context="chart html_chrome render")


def test_rq11_escape_hatch_bypasses_gate():
  """UT-RQ-11: the debug env var disables the gate even for a blank render."""
  blank = _png_bytes(Image.new("RGB", (400, 300), "white"))
  previous = os.environ.get(_ESCAPE_HATCH)
  os.environ[_ESCAPE_HATCH] = "1"
  try:
    _assert_render_quality(blank, context="chart html_chrome render")
  finally:
    if previous is None:
      os.environ.pop(_ESCAPE_HATCH, None)
    else:
      os.environ[_ESCAPE_HATCH] = previous
