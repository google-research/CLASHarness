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

import json
from pathlib import Path

from clasharness.skills.impl.email_skill import EmailSkill


def test_email_skill_preloads_scenario_emails(tmp_path: Path) -> None:
    (tmp_path / "emails.json").write_text(
        json.dumps(
            [
                {
                    "id": "pro_email_1",
                    "sender": "owner@openclaw.local",
                    "to": "agent@example.com",
                    "subject": "Launch evidence",
                    "body": "The owner evidence contains DECISION:GATED_LAUNCH_REQUIRED.",
                    "folder": "inbox",
                    "read": False,
                    "starred": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    skill = EmailSkill()
    skill.initialize(str(tmp_path), {})

    search = skill.execute("email search --query GATED_LAUNCH_REQUIRED")
    assert search.exit_code == 0
    assert "pro_email_1" in search.stdout

    read = skill.execute("email read --id pro_email_1")
    assert read.exit_code == 0
    assert "DECISION:GATED_LAUNCH_REQUIRED" in read.stdout
