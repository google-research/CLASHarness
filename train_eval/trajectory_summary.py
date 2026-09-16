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

"""Trajectory / final-response summary helpers extracted from examples/train_and_eval.py."""
from __future__ import annotations

import json
import re
from typing import Any

from clasharness.core.task import Task

from train_eval._constants import _WEATHER_CODE_TEXT


def _safe_json_dict(text: str | None) -> dict[str, Any] | None:
  raw = (text or "").strip()
  if not raw:
    return None
  try:
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else None
  except Exception:
    pass

  # Best-effort recovery when stdout has wrappers around JSON.
  start = raw.find("{")
  end = raw.rfind("}")
  if start >= 0 and end > start:
    try:
      payload = json.loads(raw[start : end + 1])
      return payload if isinstance(payload, dict) else None
    except Exception:
      return None
  return None

def _step_action(step: dict[str, Any]) -> str:
  return str(step.get("executed_action") or step.get("action") or "")

def _step_output(step: dict[str, Any]) -> str:
  clean = step.get("clean_stdout")
  if isinstance(clean, str) and clean.strip():
    return clean
  return str(step.get("stdout") or "")

def _extract_location_from_geocoding(
    trajectory: list[dict[str, Any]],
) -> str | None:
  for step in reversed(trajectory):
    action = _step_action(step)
    if "geocoding-api.open-meteo.com/v1/search" not in action:
      continue
    payload = _safe_json_dict(_step_output(step))
    if not payload:
      continue
    results = payload.get("results")
    if isinstance(results, list) and results:
      first = results[0]
      if isinstance(first, dict):
        name = first.get("name")
        if isinstance(name, str) and name:
          return name
  return None

def _extract_weather_summary(trajectory: list[dict[str, Any]]) -> str | None:
  location = _extract_location_from_geocoding(trajectory)

  for step in reversed(trajectory):
    action = _step_action(step)
    out = _step_output(step)

    if "archive-api.open-meteo.com/v1/archive" in action:
      payload = _safe_json_dict(out)
      if not payload:
        continue
      daily = (
          payload.get("daily") if isinstance(payload.get("daily"), dict) else {}
      )
      dates = daily.get("time") if isinstance(daily.get("time"), list) else []
      codes = (
          daily.get("weather_code")
          if isinstance(daily.get("weather_code"), list)
          else []
      )
      tmax = (
          daily.get("temperature_2m_max")
          if isinstance(daily.get("temperature_2m_max"), list)
          else []
      )
      tmin = (
          daily.get("temperature_2m_min")
          if isinstance(daily.get("temperature_2m_min"), list)
          else []
      )
      if not dates:
        continue
      date = dates[0]
      code = codes[0] if codes else None
      condition = (
          _WEATHER_CODE_TEXT.get(int(code), f"code {code}")
          if code is not None
          else "unknown"
      )
      high = tmax[0] if tmax else "?"
      low = tmin[0] if tmin else "?"
      place = location or "requested location"
      return (
          f"WEATHER_RESULT: {place} on {date}: {condition}, {low}°C to"
          f" {high}°C."
      )

    if "api.open-meteo.com/v1/forecast" in action:
      payload = _safe_json_dict(out)
      if not payload:
        continue
      current = (
          payload.get("current")
          if isinstance(payload.get("current"), dict)
          else {}
      )
      if not current:
        continue
      temp = current.get("temperature_2m")
      humidity = current.get("relative_humidity_2m")
      code = current.get("weather_code")
      condition = (
          _WEATHER_CODE_TEXT.get(int(code), f"code {code}")
          if code is not None
          else "unknown"
      )
      place = location or payload.get("timezone") or "requested location"
      parts = [f"WEATHER_RESULT: {place}: {condition}"]
      if temp is not None:
        parts.append(f"temp {temp}°C")
      if humidity is not None:
        parts.append(f"humidity {humidity}%")
      return ", ".join(parts) + "."

    if action.startswith("weather get "):
      line1 = re.search(
          r"Weather for (.+?) on ([0-9]{4}-[0-9]{2}-[0-9]{2}):", out
      )
      cond = re.search(r"Condition\s*:\s*(.+)", out)
      temp = re.search(r"Temp\s*:\s*([^\n]+)", out)
      hum = re.search(r"Humidity\s*:\s*([^\n]+)", out)
      if line1:
        place, date = line1.group(1), line1.group(2)
        condition = cond.group(1).strip() if cond else "unknown"
        pieces = [f"WEATHER_RESULT: {place} on {date}: {condition}"]
        if temp:
          pieces.append(f"temp {temp.group(1).strip()}")
        if hum:
          pieces.append(f"humidity {hum.group(1).strip()}")
        return ", ".join(pieces) + "."
  return None

def _extract_calendar_summary(trajectory: list[dict[str, Any]]) -> str | None:
  for step in reversed(trajectory):
    action = _step_action(step)
    out = _step_output(step)

    if "timeapi.io/api/Time/current/zone" in action:
      payload = _safe_json_dict(out)
      if not payload:
        continue
      tz = payload.get("timeZone")
      dt_value = payload.get("dateTime")
      if tz and dt_value:
        return f"CALENDAR_RESULT: Current date-time in {tz} is {dt_value}."

    if action.startswith("calendar today") or action.startswith("gcalcli now"):
      m = re.search(r"Current date-time for (.+?):\s*([^\n]+)", out)
      if m:
        return (
            f"CALENDAR_RESULT: Current date-time in {m.group(1)} is"
            f" {m.group(2)}."
        )
  return None

def _build_final_response(
    task: Task | None, trajectory: list[dict[str, Any]]
) -> str:
  if not trajectory:
    return ""
  domains = set(task.domains if task else [])
  lines: list[str] = []

  if "weather" in domains:
    weather = _extract_weather_summary(trajectory)
    if weather:
      lines.append(weather)

  if "calendar" in domains:
    calendar = _extract_calendar_summary(trajectory)
    if calendar:
      lines.append(calendar)

  # Fallback: infer from actions for mixed tasks that may omit explicit domains.
  if not lines:
    actions = " ".join(_step_action(s) for s in trajectory)
    if "weather " in actions or "open-meteo" in actions:
      weather = _extract_weather_summary(trajectory)
      if weather:
        lines.append(weather)
    if "calendar " in actions or "timeapi.io" in actions:
      calendar = _extract_calendar_summary(trajectory)
      if calendar:
        lines.append(calendar)

  return "\n".join(lines)

