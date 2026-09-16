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

"""Image + Anthropic request helpers extracted from examples/train_and_eval.py."""
from __future__ import annotations

import base64
import hashlib
import io
import re
from typing import Any

from train_eval._constants import _DOWNSCALED_IMAGE_CACHE


def _is_anthropic_temperature_unsupported(exc: Exception) -> bool:
  """True when an Anthropic error indicates `temperature` is not accepted."""
  parts = [str(exc)]
  for attr in ("message", "body"):
    val = getattr(exc, attr, None)
    if val:
      parts.append(str(val))
  msg = " ".join(parts).lower()
  if "temperature" not in msg:
    return False
  return any(
      marker in msg
      for marker in ("deprecated", "not supported", "unsupported", "not allowed")
  )

def _is_anthropic_reasoning_param_unsupported(exc: Exception) -> bool:
  """True when an Anthropic error rejects the reasoning-control parameters.

  The preferred native-reasoning request sends ``thinking={"type":"adaptive"}``
  plus a top-level ``output_config`` carrying an ``effort`` level. Older
  Vertex/Anthropic endpoints (or models that predate adaptive thinking / effort)
  reject one of those parameters with a 400 that names the offending field. We
  detect that so the caller can step down the reasoning ladder (adaptive ->
  fixed budget_tokens -> plain forced tool call) instead of failing the task.
  """
  parts = [str(exc)]
  for attr in ("message", "body"):
    val = getattr(exc, attr, None)
    if val:
      parts.append(str(val))
  msg = " ".join(parts).lower()
  names = ("output_config", "effort", "adaptive", "thinking")
  if not any(name in msg for name in names):
    return False
  markers = (
      "unsupported",
      "not supported",
      "not allowed",
      "unknown",
      "unexpected",
      "unrecognized",
      "invalid",
      "does not support",
      "deprecated",
      "not permitted",
      "not enabled",
  )
  return any(marker in msg for marker in markers)

def _anthropic_text_from_response(response: Any) -> str:
  """Concatenate text blocks from an Anthropic response.

  Reasoning models return content lists that may start with `thinking` /
  `redacted_thinking` / `tool_use` blocks, so we cannot assume `content[0]` is
  text; we select only the `text` blocks (tolerating dict- or object-shaped
  blocks, and simple fakes that expose `.text` without a `.type`).
  """
  content = getattr(response, "content", None)
  if content is None and isinstance(response, dict):
    content = response.get("content")
  parts: list[str] = []
  for block in content or []:
    if isinstance(block, dict):
      btype = block.get("type")
      text = block.get("text")
    else:
      btype = getattr(block, "type", None)
      text = getattr(block, "text", None)
    is_text = btype == "text" or (btype is None and text is not None)
    if is_text and text:
      parts.append(text)
  return "".join(parts).strip()

def _is_anthropic_image_dimension_error(exc: Exception) -> bool:
  """True when an Anthropic error indicates an image is too large (pixels)."""
  parts = [str(exc)]
  for attr in ("message", "body"):
    val = getattr(exc, attr, None)
    if val:
      parts.append(str(val))
  msg = " ".join(parts).lower()
  if "image" not in msg or "pixel" not in msg:
    return False
  return any(
      marker in msg
      for marker in (
          "dimension",
          "many-image",
          "max allowed size",
          "maximum allowed size",
          "too large",
          "exceed",
      )
  )

def _parse_anthropic_image_pixel_limit(exc: Exception) -> int | None:
  """Extract the allowed image long-edge (pixels) from an Anthropic error.

  The message typically names both the offending dimension and the cap (e.g.
  "3000 pixels ... maximum 2576 pixels"); the cap is the smaller value, so we
  take the minimum reasonable number that precedes "pixel".
  """
  parts = [str(exc)]
  for attr in ("message", "body"):
    val = getattr(exc, attr, None)
    if val:
      parts.append(str(val))
  msg = " ".join(parts)
  candidates = [
      int(m) for m in re.findall(r"(\d{2,5})\s*pixel", msg, flags=re.IGNORECASE)
  ]
  candidates = [c for c in candidates if 256 <= c <= 20000]
  if not candidates:
    return None
  return min(candidates)

def _pil_format_for_media_type(media_type: str) -> str:
  """Map an image media type to a PIL save format."""
  mt = (media_type or "").lower()
  if "jpeg" in mt or "jpg" in mt:
    return "JPEG"
  if "webp" in mt:
    return "WEBP"
  if "gif" in mt:
    return "GIF"
  return "PNG"

def _downscale_image_b64(b64_data: str, media_type: str, max_edge: int) -> str:
  """Return base64 for an image shrunk so its long edge is <= max_edge.

  Returns the input unchanged if it already fits, if Pillow is unavailable, or
  if anything goes wrong - image downscaling must never break a request. Results
  are memoized by (source hash, max_edge).
  """
  if not b64_data or max_edge <= 0:
    return b64_data
  cache_key = (hashlib.sha1(b64_data.encode("utf-8")).hexdigest(), max_edge)
  cached = _DOWNSCALED_IMAGE_CACHE.get(cache_key)
  if cached is not None:
    return cached
  try:
    from PIL import Image  # Imported lazily; optional dependency.

    raw = base64.b64decode(b64_data)
    with Image.open(io.BytesIO(raw)) as img:
      width, height = img.size
      if max(width, height) <= max_edge:
        _DOWNSCALED_IMAGE_CACHE[cache_key] = b64_data
        return b64_data
      scale = max_edge / float(max(width, height))
      new_size = (
          max(1, int(round(width * scale))),
          max(1, int(round(height * scale))),
      )
      resized = img.resize(new_size, Image.LANCZOS)
      fmt = _pil_format_for_media_type(media_type)
      if fmt == "JPEG" and resized.mode in ("RGBA", "LA", "P"):
        # JPEG has no alpha channel; flatten onto white.
        background = Image.new("RGB", resized.size, (255, 255, 255))
        rgba = resized.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        resized = background
      buffer = io.BytesIO()
      resized.save(buffer, format=fmt)
      new_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
  except Exception:  # pragma: no cover - never break a request on shrink failure
    return b64_data
  _DOWNSCALED_IMAGE_CACHE[cache_key] = new_b64
  return new_b64

def _normalize_anthropic_max_image_edge(value: str | int | None) -> str | int:
  """Normalize a --anthropic-max-image-edge value to 'auto' or a positive int.

  'auto' (and 'max'/empty/unparseable) means: start at max fidelity and learn
  the server's cap from the first oversized-image rejection. A positive integer
  forces a fixed long-edge cap.
  """
  if value is None:
    return "auto"
  if isinstance(value, bool):  # guard: bool is a subclass of int
    return "auto"
  if isinstance(value, int):
    return value if value > 0 else "auto"
  text = str(value).strip().lower()
  if text in ("", "auto", "max"):
    return "auto"
  try:
    parsed = int(text)
  except ValueError:
    return "auto"
  return parsed if parsed > 0 else "auto"

def _to_anthropic_messages(
    messages: list[dict[str, Any]],
    max_image_edge: int | None = None,
) -> list[dict[str, Any]]:
  """Convert OpenAI-shaped messages to Anthropic-compatible messages.

  The agent stores multimodal content in OpenAI form
  (``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<b64>"}}``),
  which the Anthropic Messages API rejects. This rewrites image parts into
  Anthropic image blocks (``{"type": "image", "source": {...}}``) - base64 for
  ``data:`` URLs, URL source otherwise - and passes text/string content through.
  Used for both the direct Anthropic client and AnthropicVertex (same schema).

  When ``max_image_edge`` is set, inline (``data:``) images are pre-shrunk so
  their long edge is <= that many pixels, which avoids Anthropic's oversized-
  image 400s on many-image requests.
  """
  converted: list[dict[str, Any]] = []
  for msg in messages:
    content = msg.get("content")
    if not isinstance(content, list):
      converted.append(msg)
      continue
    new_parts: list[dict[str, Any]] = []
    for part in content:
      if not isinstance(part, dict):
        continue
      ptype = part.get("type")
      if ptype == "text":
        new_parts.append({"type": "text", "text": part.get("text", "")})
      elif ptype == "image":
        # Already Anthropic-shaped; keep as-is.
        new_parts.append(part)
      elif ptype in ("tool_use", "tool_result", "thinking", "redacted_thinking"):
        # Native tool-calling turns (produced by
        # ``tool_calling.to_anthropic_messages``) are already Anthropic-shaped.
        # Keep them verbatim -- dropping them here empties the assistant/user
        # turn and Vertex/Anthropic rejects it with "messages.N: <role> messages
        # must have non-empty content". ``thinking`` / ``redacted_thinking``
        # blocks (extended-thinking reasoning replayed before a ``tool_use``)
        # must likewise survive: stripping them breaks the required thinking
        # preservation and the API rejects the turn (HTTP 400).
        new_parts.append(part)
      elif ptype == "image_url":
        url = ""
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
          url = str(image_url.get("url", ""))
        elif isinstance(image_url, str):
          url = image_url
        if not url:
          continue
        if url.startswith("data:"):
          header, _, b64_data = url.partition(",")
          media_type = header.split(";")[0].split(":")[-1] or "image/png"
          if max_image_edge:
            b64_data = _downscale_image_b64(
                b64_data, media_type, max_image_edge
            )
          new_parts.append({
              "type": "image",
              "source": {
                  "type": "base64",
                  "media_type": media_type,
                  "data": b64_data,
              },
          })
        else:
          new_parts.append(
              {"type": "image", "source": {"type": "url", "url": url}}
          )
    converted.append({"role": msg.get("role", "user"), "content": new_parts})
  return converted


def _estimate_anthropic_request_bytes(messages: list[dict[str, Any]]) -> int:
  """Rough estimate of an Anthropic request's serialized payload size in bytes.

  Sums the base64 image payloads (which dominate) plus text/URL lengths. Used to
  proactively downscale images BEFORE Vertex rejects the request with a 400
  "request size ... exceeds 30.000MB limit" -- a total-payload limit that the
  per-image pixel-cap healing never triggers on. Overestimates slightly (base64
  chars ~= wire bytes, ignoring JSON key overhead), which is the safe direction.
  """
  total = 0
  for msg in messages:
    content = msg.get("content")
    if isinstance(content, str):
      total += len(content)
      continue
    if not isinstance(content, list):
      continue
    for part in content:
      if not isinstance(part, dict):
        continue
      if part.get("type") == "text":
        total += len(part.get("text") or "")
      elif part.get("type") == "image":
        source = part.get("source") or {}
        data = source.get("data")
        if isinstance(data, str):
          total += len(data)
        url = source.get("url")
        if isinstance(url, str):
          total += len(url)
      elif part.get("type") == "tool_result":
        # Native tool-result observations carry their payload in ``content``
        # (usually a string; occasionally a list of text blocks). Count it so
        # tool-heavy native histories are not understated against the budget.
        tr_content = part.get("content")
        if isinstance(tr_content, str):
          total += len(tr_content)
        elif isinstance(tr_content, list):
          for block in tr_content:
            if isinstance(block, dict) and block.get("type") == "text":
              total += len(block.get("text") or "")
  return total


def _fit_anthropic_messages_to_byte_budget(
    base_messages: list[dict[str, Any]],
    max_image_edge: int | None,
    byte_budget: int,
    edge_ladder: tuple[int, ...],
) -> tuple[list[dict[str, Any]], int | None]:
  """Build Anthropic messages, shrinking images until they fit ``byte_budget``.

  Tries the least-aggressive option first (the current ``max_image_edge`` cap,
  or originals when it is ``None`` so ``auto`` keeps max fidelity), then descends
  ``edge_ladder``, stopping at the first candidate whose estimated request size
  is within budget. If even the smallest edge does not fit, returns the smallest
  attempt (best effort) -- the request still goes out, and a residual oversize is
  classified as a provider failure rather than charged to the model.

  Returns ``(converted_messages, edge_used)``.
  """
  candidates: list[int | None] = [max_image_edge]
  for edge in edge_ladder:
    if max_image_edge is not None and edge >= max_image_edge:
      continue
    candidates.append(edge)

  converted: list[dict[str, Any]] = []
  edge_used = max_image_edge
  for cand in candidates:
    converted = _to_anthropic_messages(base_messages, max_image_edge=cand)
    edge_used = cand
    if _estimate_anthropic_request_bytes(converted) <= byte_budget:
      return converted, edge_used
  return converted, edge_used

