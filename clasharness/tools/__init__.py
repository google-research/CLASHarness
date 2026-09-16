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

"""Tool specification helpers for CLASHarness command surfaces."""

from clasharness.tools.agent_tool_spec import (
    AgentCommandSpec,
    AgentToolFamily,
    AgentToolSpec,
    ToolSpecViolation,
    allowed_prefixes,
    cli_verbs_by_family,
    iter_command_specs,
    load_agent_tool_spec,
    normalize_command_quoting,
    required_flags_by_tokens,
    required_positionals_by_tokens,
    validate_command_against_spec,
    validate_task_solution_commands,
)

__all__ = [
    "AgentCommandSpec",
    "AgentToolFamily",
    "AgentToolSpec",
    "ToolSpecViolation",
    "allowed_prefixes",
    "cli_verbs_by_family",
    "iter_command_specs",
    "load_agent_tool_spec",
    "normalize_command_quoting",
    "required_flags_by_tokens",
    "required_positionals_by_tokens",
    "validate_command_against_spec",
    "validate_task_solution_commands",
]
