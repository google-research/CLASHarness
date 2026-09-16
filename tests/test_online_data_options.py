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

from __future__ import annotations

from clasharness.backend import calendar_backend as calendar_mod
from clasharness.backend import email_backend as email_mod
from clasharness.backend import tasks_backend as tasks_mod
from clasharness.backend import weather_backend as weather_mod
from clasharness.backend.calendar_backend import CalendarBackend
from clasharness.backend.email_backend import EmailBackend
from clasharness.backend.tasks_backend import TasksBackend
from clasharness.backend.weather_backend import WeatherBackend


def test_weather_online_option_disabled_uses_mock(tmp_path):
    backend = WeatherBackend()
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli("weather get --location 'New York' --online")
    assert result.exit_code == 0
    assert "Condition" in result.stdout
    assert "Source    : mock" in result.stdout


def test_weather_online_option_enabled_returns_source(tmp_path):
    backend = WeatherBackend()
    backend.initialize(str(tmp_path), {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"})
    result = backend.execute_cli("weather get --location 'New York' --online")
    assert result.exit_code == 0
    assert "Condition" in result.stdout
    assert "Source" in result.stdout


def test_weather_auto_online_by_env_without_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(weather_mod, "_online_today_date", lambda: "2030-12-31")
    monkeypatch.setattr(weather_mod, "_get_weather_online", lambda location, date: None)
    backend = WeatherBackend()
    backend.initialize(str(tmp_path), {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"})
    result = backend.execute_cli("weather get --location 'New York'")
    assert result.exit_code == 0
    assert "on 2030-12-31" in result.stdout
    assert "Source    : mock-fallback" in result.stdout


def test_weather_online_without_date_uses_current_date(monkeypatch, tmp_path):
    monkeypatch.setattr(weather_mod, "_online_today_date", lambda: "2030-12-31")
    monkeypatch.setattr(weather_mod, "_get_weather_online", lambda location, date: None)
    backend = WeatherBackend()
    backend.initialize(str(tmp_path), {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"})
    result = backend.execute_cli("weather get --location 'New York' --online")
    assert result.exit_code == 0
    assert "on 2030-12-31" in result.stdout
    assert "Source    : mock-fallback" in result.stdout


def test_calendar_today_online_disabled_uses_mock(tmp_path):
    backend = CalendarBackend()
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli("calendar today --timezone UTC --online")
    assert result.exit_code == 0
    assert "Current date-time for UTC:" in result.stdout
    assert "Source: mock" in result.stdout


def test_calendar_today_online_enabled_returns_source(tmp_path):
    backend = CalendarBackend()
    backend.initialize(str(tmp_path), {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"})
    result = backend.execute_cli("calendar today --timezone UTC --online")
    assert result.exit_code == 0
    assert "Current date-time for UTC:" in result.stdout
    assert "Source:" in result.stdout


def test_calendar_today_rejects_template_timezone(tmp_path):
    backend = CalendarBackend()
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli("calendar today --timezone TIMEZONE")
    assert result.exit_code == 1
    assert "Invalid timezone value: TIMEZONE" in result.stderr


def test_calendar_today_auto_online_by_env_without_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "clasharness.backend.calendar_backend._get_online_time",
        lambda timezone: "2030-12-31T12:34:56",
    )
    backend = CalendarBackend()
    backend.initialize(str(tmp_path), {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"})
    result = backend.execute_cli("calendar today --timezone UTC")
    assert result.exit_code == 0
    assert "Current date-time for UTC: 2030-12-31T12:34:56" in result.stdout
    assert "Source: online" in result.stdout


def test_gcalcli_now_online_enabled_returns_source(tmp_path):
    backend = CalendarBackend()
    backend.initialize(str(tmp_path), {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"})
    result = backend.execute_cli("gcalcli now --timezone UTC --online")
    assert result.exit_code == 0
    assert "Current date-time for UTC:" in result.stdout
    assert "Source:" in result.stdout


def test_weather_online_strict_mode_disables_mock_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(weather_mod, "_get_weather_online", lambda location, date: None)
    monkeypatch.setattr(weather_mod, "_online_today_date", lambda: "2030-12-31")
    backend = WeatherBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
        },
    )
    result = backend.execute_cli("weather get --location 'New York' --online")
    assert result.exit_code == 1
    assert "Strict online mode is enabled" in result.stderr


def test_weather_forecast_rejects_template_values(tmp_path):
    backend = WeatherBackend()
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli("weather forecast --location LOCATION --days N")
    assert result.exit_code == 1
    assert "Invalid location value: LOCATION" in result.stderr


def test_calendar_today_online_strict_mode_disables_mock_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "clasharness.backend.calendar_backend._get_online_time",
        lambda timezone: None,
    )
    backend = CalendarBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
        },
    )
    result = backend.execute_cli("calendar today --timezone UTC --online")
    assert result.exit_code == 1
    assert "Strict online mode is enabled" in result.stderr


def test_gcalcli_now_online_strict_mode_disables_mock_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "clasharness.backend.calendar_backend._get_online_time",
        lambda timezone: None,
    )
    backend = CalendarBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
        },
    )
    result = backend.execute_cli("gcalcli now --timezone UTC --online")
    assert result.exit_code == 1
    assert "Strict online mode is enabled" in result.stderr


def test_calendar_add_event_online_attempts_provider(monkeypatch, tmp_path):
    called: list[list[str]] = []

    monkeypatch.setattr(
        calendar_mod,
        "_resolve_calendar_provider",
        lambda preference: "gcalcli",
    )
    monkeypatch.setattr(
        calendar_mod,
        "_run_calendar_provider_command",
        lambda cmd, timeout: (called.append(cmd) or (0, "ok", "")),
    )

    backend = CalendarBackend()
    backend.initialize(str(tmp_path), {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"})
    result = backend.execute_cli("calendar add-event --title 'Team standup' --start 2026-03-10T09:00")
    assert result.exit_code == 0
    assert called
    assert called[0][0:2] == ["gcalcli", "add"]


def test_calendar_add_event_online_strict_requires_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calendar_mod,
        "_resolve_calendar_provider",
        lambda preference: None,
    )
    backend = CalendarBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
        },
    )
    result = backend.execute_cli("calendar add-event --title 'Team standup' --start 2026-03-10T09:00")
    assert result.exit_code == 1
    assert "Strict online mode is enabled" in result.stderr


def test_calendar_add_event_google_api_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calendar_mod,
        "_resolve_calendar_provider",
        lambda preference: "google_api",
    )
    monkeypatch.setattr(
        "clasharness.skills.impl.calendar_skill._run_google_api_action",
        lambda action, kwargs, settings: (
            0,
            '{"id":"evt_google_1"}',
            "",
            [
                {
                    "action": "google_api.events.insert",
                    "stdout": '{"id":"evt_google_1"}',
                    "stderr": "",
                    "exit_code": 0,
                }
            ],
        ),
    )
    backend = CalendarBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_CALENDAR_PROVIDER": "google_api",
        },
    )
    result = backend.execute_cli("calendar add-event --title 'Team standup' --start 2026-03-10T09:00")
    assert result.exit_code == 0
    assert result.execution_trace
    assert result.execution_trace[0]["action"].startswith("google_api.events.")


def test_calendar_default_provider_prefers_google_api(monkeypatch, tmp_path):
    captured: dict[str, str] = {}

    def _resolve(preference: str):
        captured["preference"] = preference
        return "google_api"

    monkeypatch.setattr(calendar_mod, "_resolve_calendar_provider", _resolve)
    monkeypatch.setattr(
        "clasharness.skills.impl.calendar_skill._run_google_api_action",
        lambda action, kwargs, settings: (0, "{}", "", []),
    )

    backend = CalendarBackend()
    backend.initialize(
        str(tmp_path),
        {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"},
    )
    result = backend.execute_cli("calendar add-event --title 'Team standup' --start 2026-03-10T09:00")
    assert result.exit_code == 0
    assert captured.get("preference") == "google_api"


def test_email_online_strict_mode_requires_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        email_mod,
        "_run_email_online_action",
        lambda **kwargs: (
            False,
            "provider unavailable",
            [
                {
                    "action": "email.online",
                    "provider": "none",
                    "stdout": "",
                    "stderr": "provider unavailable",
                    "exit_code": 1,
                }
            ],
            {},
        ),
    )
    backend = EmailBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
        },
    )
    result = backend.execute_cli(
        "email send --to bob@example.com --subject 'Hello' --body 'Test'"
    )
    assert result.exit_code == 1
    assert "Strict online mode is enabled" in result.stderr


def test_email_online_non_strict_allows_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        email_mod,
        "_run_email_online_action",
        lambda **kwargs: (
            False,
            "provider unavailable",
            [
                {
                    "action": "email.online",
                    "provider": "none",
                    "stdout": "",
                    "stderr": "provider unavailable",
                    "exit_code": 1,
                }
            ],
            {},
        ),
    )
    backend = EmailBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "0",
        },
    )
    result = backend.execute_cli(
        "email send --to bob@example.com --subject 'Hello' --body 'Test'"
    )
    assert result.exit_code == 0
    assert "Email sent to bob@example.com" in result.stdout
    assert result.execution_trace


def test_email_read_returns_subject_and_body(tmp_path):
    backend = EmailBackend()
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli("email read --id email_seed_3")
    assert result.exit_code == 0
    assert "Subject: Budget report" in result.stdout
    assert "Body:" in result.stdout
    assert "budget report for this quarter" in result.stdout.lower()


def test_email_seed_id_mark_still_works_in_strict_online_mode(tmp_path):
    backend = EmailBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
        },
    )
    result = backend.execute_cli("email mark --id email_seed_1 --flag read")
    assert result.exit_code == 0
    assert "marked as read" in result.stdout
    assert result.execution_trace
    assert any(t.get("reason") == "unmapped_seed_id" for t in result.execution_trace)


def test_email_seed_id_read_still_works_in_strict_online_mode(tmp_path):
    backend = EmailBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
        },
    )
    result = backend.execute_cli("email read --id email_seed_1")
    assert result.exit_code == 0
    assert "Subject: Project proposal" in result.stdout
    assert result.execution_trace
    assert any(t.get("reason") == "unmapped_seed_id" for t in result.execution_trace)


def test_email_read_uses_online_mapping_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(
        email_mod,
        "_run_email_online_action",
        lambda **kwargs: (
            True,
            "",
            [
                {
                    "action": "google_api.gmail.messages.read",
                    "provider": "google_api",
                    "stdout": "{}",
                    "stderr": "",
                    "exit_code": 0,
                }
            ],
            {
                "email": {
                    "id": "remote_1",
                    "sender": "alice@example.com",
                    "to": "me@example.com",
                    "subject": "Remote proposal",
                    "body": "Remote body text",
                    "folder": "inbox",
                    "read": True,
                    "starred": False,
                }
            },
        ),
    )
    backend = EmailBackend()
    backend.initialize(
        str(tmp_path),
        {"CLASHARNESS_ENABLE_ONLINE_DATA": "1"},
    )
    backend._emails.append(  # noqa: SLF001 - compatibility accessor
        {
            "id": "email_9999",
            "sender": "me@example.com",
            "to": "alice@example.com",
            "subject": "Local draft",
            "body": "Local body",
            "folder": "sent",
            "read": True,
            "starred": False,
        }
    )
    backend._skill._online_ids_by_local_id["email_9999"] = "remote_1"  # noqa: SLF001
    result = backend.execute_cli("email read --id email_9999")
    assert result.exit_code == 0
    assert "Subject: Remote proposal" in result.stdout
    assert "Remote body text" in result.stdout
    assert result.execution_trace


def test_tasks_online_strict_mode_requires_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tasks_mod,
        "_run_tasks_online_action",
        lambda **kwargs: (
            False,
            "provider unavailable",
            [
                {
                    "action": "tasks.online",
                    "provider": "none",
                    "stdout": "",
                    "stderr": "provider unavailable",
                    "exit_code": 1,
                }
            ],
            {},
        ),
    )
    backend = TasksBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
            "CLASHARNESS_TASKS_PROVIDER": "google_api",
        },
    )
    result = backend.execute_cli("tasks add --title 'Write report' --due 2026-03-10")
    assert result.exit_code == 1
    assert "Strict online mode is enabled" in result.stderr


def test_tasks_online_non_strict_allows_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tasks_mod,
        "_run_tasks_online_action",
        lambda **kwargs: (
            False,
            "provider unavailable",
            [
                {
                    "action": "tasks.online",
                    "provider": "none",
                    "stdout": "",
                    "stderr": "provider unavailable",
                    "exit_code": 1,
                }
            ],
            {},
        ),
    )
    backend = TasksBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "0",
            "CLASHARNESS_TASKS_PROVIDER": "google_api",
        },
    )
    result = backend.execute_cli("tasks add --title 'Write report' --due 2026-03-10")
    assert result.exit_code == 0
    assert "Task created:" in result.stdout
    assert result.execution_trace


def test_tasks_seed_complete_still_works_in_strict_online_mode(tmp_path):
    backend = TasksBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "1",
            "CLASHARNESS_TASKS_PROVIDER": "google_api",
        },
    )
    result = backend.execute_cli("tasks complete --id task_seed_1")
    assert result.exit_code == 0
    assert "marked as done" in result.stdout
    assert result.execution_trace
    assert any(t.get("reason") == "unmapped_seed_id" for t in result.execution_trace)


def test_tasks_default_provider_auto_attempts_online(monkeypatch, tmp_path):
    captured: dict[str, str] = {}

    def _fake_run(**kwargs):
        captured["provider_preference"] = str(kwargs.get("provider_preference"))
        return (
            False,
            "provider unavailable",
            [
                {
                    "action": "tasks.online",
                    "provider": "none",
                    "stdout": "",
                    "stderr": "provider unavailable",
                    "exit_code": 1,
                }
            ],
            {},
        )

    monkeypatch.setattr(tasks_mod, "_run_tasks_online_action", _fake_run)
    backend = TasksBackend()
    backend.initialize(
        str(tmp_path),
        {
            "CLASHARNESS_ENABLE_ONLINE_DATA": "1",
            "CLASHARNESS_STRICT_ONLINE_DATA": "0",
        },
    )
    result = backend.execute_cli("tasks add --title 'Local fallback task' --due 2026-03-10")
    assert captured.get("provider_preference") == "auto"
    assert result.exit_code == 0
    assert "Task created:" in result.stdout
    assert result.execution_trace
