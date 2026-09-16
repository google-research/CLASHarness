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

"""Layer-2a pure-helper tests for the external-network skills.

Phase 3 context
---------------
The five external skills (calendar, curl, email, tasks, weather) have a *mock*
path (deterministic simulation, the eval default) and a *real* path that talks to
the network. The real path itself cannot run hermetically, but each skill (except
curl) factors its real path into small **pure helpers** — query builders,
datetime/format normalizers, provider resolution, and response parsers — that are
fully deterministic and network-free. This module locks that pure logic down so a
regression in the real path is caught without any live calls.

  * ``curl`` has no separable pure logic: its real path is a raw ``subprocess``
    passthrough, so it is intentionally *not* covered here. It is exercised by its
    existing mock tests and by the opt-in live suite (Layer 2b).
  * The live end-to-end behavior of these helpers (actual HTTP) is Layer 2b.

Everything here is hermetic: the one helper that would shell out
(``weather._get_weather_online``) is tested by swapping its module-level
``_run_json_cli`` for a canned responder, so no network or ``curl`` binary is
touched.

Skips (none needed here) would use ``pytest.skip()`` at call time to behave under
both a bare-``unittest`` runner and pytest.
"""

from __future__ import annotations

import base64
import contextlib

from clasharness.skills.impl import calendar_skill, email_skill, tasks_skill, weather_skill


@contextlib.contextmanager
def _swap(module, name, value):
    """Temporarily replace a module-level attribute (no monkeypatch fixture dep)."""
    sentinel = object()
    original = getattr(module, name, sentinel)
    setattr(module, name, value)
    try:
        yield
    finally:
        if original is sentinel:
            delattr(module, name)
        else:
            setattr(module, name, original)


def _b64url(text: str) -> str:
    """Gmail-style unpadded urlsafe base64 (exercises the padding logic)."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


# --------------------------------------------------------------------------- #
# weather                                                                       #
# --------------------------------------------------------------------------- #
def test_weather_code_table_maps_known_and_unknown():
    assert weather_skill._WEATHER_CODE_TO_TEXT[0] == "clear"
    assert weather_skill._WEATHER_CODE_TO_TEXT.get(999, "unknown") == "unknown"


def test_weather_hash_is_deterministic():
    out = weather_skill._hash_weather("Paris", "2026-01-01")
    assert out == {"condition": "cloudy", "temp_c": 27, "humidity": 87}
    # Same inputs -> same output (the mock path relies on this stability).
    assert weather_skill._hash_weather("Paris", "2026-01-01") == out


def test_weather_get_prefers_scenario_then_falls_back_to_hash():
    # A seeded scenario key returns the canned value.
    assert weather_skill._get_weather("london", "2026-03-01") == {
        "condition": "rainy",
        "temp_c": 6,
        "humidity": 90,
    }
    # An unseeded key falls back to the deterministic hash.
    assert weather_skill._get_weather("Nowhere", "1999-12-31") == weather_skill._hash_weather(
        "Nowhere", "1999-12-31"
    )


def test_weather_looks_like_placeholder_matches_fixed_token_set_only():
    assert weather_skill._looks_like_placeholder("LOCATION") is True
    assert weather_skill._looks_like_placeholder("[DATE]") is True  # stripped of brackets
    assert weather_skill._looks_like_placeholder("Paris") is False
    # NOTE: this is a fixed-token detector, not a generic one: arbitrary
    # ALL-CAPS tokens like YOUR_KEY are *not* treated as placeholders.
    assert weather_skill._looks_like_placeholder("YOUR_KEY") is False


def _canned_weather_cli(*, geo_results=None, current=None, daily=None):
    """Build a fake ``_run_json_cli`` dispatching by URL (cmd[2])."""

    def fake(cmd, *, timeout, trace, required=True):
        url = cmd[2]
        trace.append({"action": " ".join(cmd), "stdout": "", "stderr": "", "exit_code": 0})
        if "geocoding-api" in url:
            return {"results": geo_results if geo_results is not None else []}
        if "timeapi.io" in url:
            return {}
        if "forecast" in url:
            return {"current": current} if current is not None else {}
        if "archive" in url:
            return {"daily": daily} if daily is not None else {}
        return {}

    return fake


def test_weather_online_archive_branch_parses_historical_payload():
    fake = _canned_weather_cli(
        geo_results=[{"latitude": 48.85, "longitude": 2.35, "timezone": "Europe/Paris"}],
        daily={
            "weather_code": [3],
            "temperature_2m_max": [10.0],
            "temperature_2m_min": [4.0],
        },
    )
    with _swap(weather_skill, "_run_json_cli", fake):
        # A date far in the past never equals "today", so the archive branch runs.
        out = weather_skill._get_weather_online("Paris", "2020-01-01")
    assert out == {"condition": "overcast", "temp_c": 7, "humidity": 55}


def test_weather_online_forecast_branch_parses_current_payload():
    fake = _canned_weather_cli(
        geo_results=[{"latitude": 48.85, "longitude": 2.35, "timezone": "Europe/Paris"}],
        current={"weather_code": 0, "temperature_2m": 21.4, "relative_humidity_2m": 60},
    )
    with _swap(weather_skill, "_run_json_cli", fake), _swap(
        weather_skill, "_online_today_date", lambda: "2026-06-15"
    ):
        out = weather_skill._get_weather_online("Paris", "2026-06-15")
    assert out == {"condition": "clear", "temp_c": 21, "humidity": 60}


def test_weather_online_returns_none_when_geocoding_empty():
    fake = _canned_weather_cli(geo_results=[])
    with _swap(weather_skill, "_run_json_cli", fake):
        assert weather_skill._get_weather_online("Nowhereville", "2020-01-01") is None


# --------------------------------------------------------------------------- #
# calendar                                                                      #
# --------------------------------------------------------------------------- #
def test_calendar_resolve_provider_deterministic_prefs():
    assert calendar_skill._resolve_calendar_provider("mock") is None
    assert calendar_skill._resolve_calendar_provider("google_api") == "google_api"
    assert calendar_skill._resolve_calendar_provider("  MOCK ") is None  # trimmed + lowered


def test_calendar_build_gcalcli_add_command():
    cmd = calendar_skill._build_calendar_provider_command(
        "gcalcli", "add", title="Sync", start="2026-02-01T10:00:00"
    )
    assert cmd == ["gcalcli", "add", "--title", "Sync", "--when", "2026-02-01 10:00"]


def test_calendar_build_gcalcli_add_with_optional_fields():
    cmd = calendar_skill._build_calendar_provider_command(
        "gcalcli",
        "add",
        title="Sync",
        start="2026-02-01T10:00:00",
        end="2026-02-01T11:00:00",
        location="HQ",
        attendees=["a@b.com", "c@d.com"],
    )
    assert cmd == [
        "gcalcli",
        "add",
        "--title",
        "Sync",
        "--when",
        "2026-02-01 10:00",
        "--end",
        "2026-02-01 11:00",
        "--where",
        "HQ",
        "--who",
        "a@b.com,c@d.com",
    ]


def test_calendar_build_khal_add_infers_one_hour_end():
    cmd = calendar_skill._build_calendar_provider_command(
        "khal", "add", title="Sync", start="2026-02-01T10:00:00"
    )
    assert cmd == ["khal", "new", "2026-02-01", "10:00", "11:00", "Sync"]


def test_calendar_build_command_guards_missing_title_and_unknown_provider():
    assert (
        calendar_skill._build_calendar_provider_command(
            "gcalcli", "add", start="2026-02-01T10:00:00"
        )
        is None
    )
    assert (
        calendar_skill._build_calendar_provider_command(
            "nope", "add", title="x", start="2026-02-01T10:00:00"
        )
        is None
    )


def test_calendar_format_cli_datetime_normalizes_and_passes_through():
    assert calendar_skill._format_cli_datetime("2026-02-01T10:00:00") == "2026-02-01 10:00"
    # Unparseable input is returned verbatim rather than raising.
    assert calendar_skill._format_cli_datetime("not-a-date") == "not-a-date"


def test_calendar_parse_iso_datetime():
    parsed = calendar_skill._parse_iso_datetime("2026-02-01T10:00:00")
    assert parsed is not None and parsed.year == 2026 and parsed.hour == 10
    assert calendar_skill._parse_iso_datetime("") is None
    assert calendar_skill._parse_iso_datetime("bad") is None


def test_calendar_build_google_settings_defaults():
    settings = calendar_skill._build_google_settings({})
    assert settings["calendar_id"] == "primary"
    assert settings["timezone"] == "UTC"
    assert settings["scopes"] == ["https://www.googleapis.com/auth/calendar"]
    assert settings["token_file"].endswith("google_calendar_token.json")


# --------------------------------------------------------------------------- #
# email                                                                         #
# --------------------------------------------------------------------------- #
def test_email_build_gmail_query_combines_filters():
    q = email_skill._build_gmail_query(
        {"folder": "inbox", "unread_only": True, "from_filter": "a@b.com", "query": "report"}
    )
    assert q == "in:inbox is:unread from:a@b.com report"


def test_email_build_gmail_query_labels_and_empty():
    assert email_skill._build_gmail_query({"folder": "work"}) == "label:work"
    assert email_skill._build_gmail_query({}) == ""


def test_email_decode_body_data_handles_unpadded_urlsafe_b64():
    assert email_skill._gmail_decode_body_data(_b64url("line1\nline2")) == "line1\nline2"


def test_email_extract_body_prefers_plain_over_html_in_multipart():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64url("plain part")}},
            {"mimeType": "text/html", "body": {"data": _b64url("<p>html</p>")}},
        ],
    }
    assert email_skill._gmail_extract_body_text(payload) == "plain part"


def test_email_gmail_message_to_email_parses_headers_labels_and_body():
    message = {
        "id": "m1",
        "labelIds": ["INBOX", "STARRED"],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "a@b.com"},
                {"name": "To", "value": "me@x.com"},
                {"name": "Subject", "value": "Hi"},
            ],
            "body": {"data": _b64url("Hello body")},
        },
    }
    email = email_skill._gmail_message_to_email(message)
    assert email["id"] == "m1"
    assert email["sender"] == "a@b.com"
    assert email["to"] == "me@x.com"
    assert email["subject"] == "Hi"
    assert email["body"] == "Hello body"
    assert email["folder"] == "inbox"
    assert email["read"] is True  # no UNREAD label
    assert email["starred"] is True


def test_email_gmail_message_to_email_marks_unread():
    message = {
        "id": "m2",
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {"headers": [], "body": {}},
    }
    email = email_skill._gmail_message_to_email(message)
    assert email["read"] is False
    assert email["starred"] is False


def test_email_resolve_provider_mock_is_none():
    assert email_skill._resolve_email_provider("mock", {}, {}) is None


def test_email_parse_himalaya_read_output_splits_headers_and_body():
    parsed = email_skill._parse_himalaya_read_output(
        "id9", "From: a@b.com\nSubject: Yo\n\nline1\nline2"
    )
    assert parsed["id"] == "id9"
    assert parsed["sender"] == "a@b.com"
    assert parsed["subject"] == "Yo"
    assert parsed["body"] == "line1\nline2"


# --------------------------------------------------------------------------- #
# tasks                                                                         #
# --------------------------------------------------------------------------- #
def test_tasks_resolve_provider_mock_is_none():
    assert tasks_skill._resolve_tasks_provider("mock", {}) is None


def test_tasks_to_google_due_date_only():
    assert tasks_skill._to_google_due("2026-03-15") == "2026-03-15T00:00:00Z"


def test_tasks_to_google_due_datetime_gets_utc_zulu():
    assert tasks_skill._to_google_due("2026-03-15T09:30:00") == "2026-03-15T09:30:00Z"


def test_tasks_to_google_due_rejects_empty_and_garbage():
    assert tasks_skill._to_google_due("") is None
    assert tasks_skill._to_google_due("nope") is None
    assert tasks_skill._to_google_due(None) is None


def test_tasks_build_google_settings_defaults_and_overrides():
    defaults = tasks_skill._build_google_tasks_settings({})
    assert defaults["tasklist_id"] == "@default"
    assert defaults["scopes"] == ["https://www.googleapis.com/auth/tasks"]
    assert defaults["allow_interactive_auth"] is False
    assert defaults["token_file"].endswith("google_tasks_token.json")

    overridden = tasks_skill._build_google_tasks_settings(
        {
            "CLASHARNESS_GOOGLE_TASKS_LIST_ID": "XYZ",
            "CLASHARNESS_GOOGLE_TASKS_SCOPES": "s1, s2",
            "CLASHARNESS_GOOGLE_ALLOW_INTERACTIVE_AUTH": "yes",
        }
    )
    assert overridden["tasklist_id"] == "XYZ"
    assert overridden["scopes"] == ["s1", "s2"]
    assert overridden["allow_interactive_auth"] is True


if __name__ == "__main__":  # pragma: no cover - convenience for direct runs
    import sys

    sys.exit(0)
