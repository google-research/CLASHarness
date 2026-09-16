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

"""Tests for the Vertex AI image provider (mirrors the Vertex LLM path)."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from clasharness.scenario import vlm_asset_generators as gen


def _clear_vertex_env(monkeypatch):
    for var in (
        "VERTEX_PROJECT", "VERTEX_LOCATION",
        "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
        "GEMINI_PROJECT", "GEMINI_LOCATION",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolve_vertex_target_from_env(monkeypatch):
    _clear_vertex_env(monkeypatch)
    monkeypatch.setenv("VERTEX_PROJECT", "proj-from-env")
    monkeypatch.setenv("VERTEX_LOCATION", "europe-west4")
    project, location = gen._resolve_vertex_target(None, None)
    assert project == "proj-from-env"
    assert location == "europe-west4"


def test_resolve_vertex_target_precedence_and_default(monkeypatch):
    _clear_vertex_env(monkeypatch)
    # Fallback chain: GOOGLE_CLOUD_PROJECT used when VERTEX_PROJECT absent.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gcp-proj")
    project, location = gen._resolve_vertex_target(None, None)
    assert project == "gcp-proj"
    assert location == "us-central1"  # default when nothing set
    # Explicit args win over env.
    monkeypatch.setenv("VERTEX_PROJECT", "env-proj")
    project, location = gen._resolve_vertex_target("explicit-proj", "explicit-loc")
    assert project == "explicit-proj"
    assert location == "explicit-loc"


def test_extract_genai_inline_image_bytes():
    raw = b"\x89PNG\r\n\x1a\nfake"
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(inline_data=None),  # ignored
                        SimpleNamespace(
                            inline_data=SimpleNamespace(data=raw, mime_type="image/png")
                        ),
                    ]
                )
            )
        ]
    )
    data, mime = gen._extract_genai_inline_image(response)
    assert data == raw
    assert mime == "image/png"


def test_extract_genai_inline_image_base64_string():
    raw = b"hello-image-bytes"
    encoded = base64.b64encode(raw).decode("ascii")
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(data=encoded, mime_type="image/jpeg"))]
                )
            )
        ]
    )
    data, mime = gen._extract_genai_inline_image(response)
    assert data == raw
    assert mime == "image/jpeg"


def test_extract_genai_inline_image_raises_when_absent():
    response = SimpleNamespace(candidates=[])
    with pytest.raises(RuntimeError, match="did not contain inline image"):
        gen._extract_genai_inline_image(response)


def test_vertex_image_generator_generate(monkeypatch):
    _clear_vertex_env(monkeypatch)
    monkeypatch.setenv("VERTEX_PROJECT", "test-proj")

    png = b"\x89PNG\r\n\x1a\nrendered"

    class _FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, *, model, contents, config):
            self.calls.append((model, contents))
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(inline_data=SimpleNamespace(data=png, mime_type="image/png"))]
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(models=_FakeModels())

    g = gen.VertexImageGenerator(model="gemini-3-pro-image")
    # Avoid importing the real google-genai SDK / building a real client.
    monkeypatch.setattr(g, "_get_client", lambda: fake_client)
    # Stub the SDK types import used inside generate().
    import sys
    fake_types = SimpleNamespace(GenerateContentConfig=lambda **kw: dict(kw))
    monkeypatch.setitem(sys.modules, "google.genai", SimpleNamespace(types=fake_types))
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    asset = g.generate(prompt="draw a server rack", caption="rack", anchors=["a1"])
    assert asset.data == png
    assert asset.mime_type == "image/png"
    assert asset.extension == "png"
    assert asset.metadata["provider"] == "vertex"
    assert asset.metadata["model"] == "gemini-3-pro-image"
    assert asset.metadata["project"] == "test-proj"
    assert fake_client.models.calls  # the model was actually invoked


def test_vertex_image_generator_lazy_client_no_creds_needed(monkeypatch):
    """Constructing the generator must not require the SDK or credentials."""
    _clear_vertex_env(monkeypatch)
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    g = gen.VertexImageGenerator(model="gemini-3-pro-image", location="us")
    assert g.project == "p"
    assert g.location == "us"
    assert g._client is None  # client is built lazily on first generate()
