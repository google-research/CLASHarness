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

"""Concrete built-in skills."""

from clasharness.skills.impl.calendar_skill import CalendarSkill
from clasharness.skills.impl.email_skill import EmailSkill
from clasharness.skills.impl.tasks_skill import TasksSkill
from clasharness.skills.impl.weather_skill import WeatherSkill
from clasharness.skills.impl.message_skill import MessageSkill
from clasharness.skills.impl.agent_skill import AgentSkill
from clasharness.skills.impl.channel_skill import ChannelSkill
from clasharness.skills.impl.system_skill import SystemSkill
from clasharness.skills.impl.git_skill import GitSkill
from clasharness.skills.impl.jq_skill import JqSkill
from clasharness.skills.impl.grep_skill import GrepSkill
from clasharness.skills.impl.shell_execution_skill import ShellExecutionSkill
from clasharness.skills.impl.lock_skill import LockSkill
from clasharness.skills.impl.deploy_skill import DeploySkill
from clasharness.skills.impl.booking_skill import BookingSkill
from clasharness.skills.impl.pairing_skill import PairingSkill
from clasharness.skills.impl.approval_skill import ApprovalSkill
from clasharness.skills.impl.containment_skill import ContainmentSkill
from clasharness.skills.impl.review_skill import ReviewSkill

__all__ = [
    "CalendarSkill",
    "EmailSkill",
    "TasksSkill",
    "WeatherSkill",
    "MessageSkill",
    "AgentSkill",
    "ChannelSkill",
    "SystemSkill",
    "GitSkill",
    "JqSkill",
    "GrepSkill",
    "ShellExecutionSkill",
    "LockSkill",
    "DeploySkill",
    "BookingSkill",
    "PairingSkill",
    "ApprovalSkill",
    "ContainmentSkill",
    "ReviewSkill",
]


