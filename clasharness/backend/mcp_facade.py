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

"""Declarative MCP facade: the single source of truth for the curated agent API.

This module consolidates what used to be two hand-maintained, drift-prone blobs
in ``backend/mcp_server.py`` -- the ``MCP_TOOLS`` schema list and the
``call_tool`` if/elif dispatch chain -- into one declaration per tool:

* :data:`TOOL_SCHEMAS` -- the exact curated ``tools/list`` payload (name,
  description, inputSchema). This is the *agent-facing* API surface and is kept
  intentionally distinct from the raw per-command SkillSpec surface.
* :data:`_BINDINGS` / :data:`_COMPOSITE` -- how each tool's arguments are turned
  into a workspace CLI string for ``env.step``.

``mcp_server`` is now a thin driver: it calls :func:`build_tools_list` for
``tools/list`` and :func:`build_cli` for ``tools/call``. Adding or changing a
tool is a single edit here.

This file is emitted by ``scratch/gen_mcp_facade.py``; correctness is guarded by
``tests/test_mcp_facade.py``.
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Any, Callable


class DirectResult:
    """A binder result that is returned to the caller as-is (no ``env.step``).

    Used for input-validation errors (e.g. an unsupported file extension) that
    the legacy dispatch returned directly as tool content.
    """

    __slots__ = ("content",)

    def __init__(self, content: list[dict[str, Any]]):
        self.content = content


# --------------------------------------------------------------------------- #
# Composite binders (imperative). Signature: (arguments, get_channel_url) ->
# str | DirectResult.
# --------------------------------------------------------------------------- #
def _bind_write_file(a, get_channel_url):
    content = shlex.quote(a["content"])
    path = shlex.quote(a["path"])
    return f"echo {content} > {path}"


def _bind_append_file(a, get_channel_url):
    content = shlex.quote(a["content"])
    path = shlex.quote(a["path"])
    return f"echo {content} >> {path}"


def _bind_send_message(a, get_channel_url):
    url = get_channel_url(a["channel"])
    if a["channel"].lower() == "telegram":
        payload = json.dumps({"chat_id": a["target"], "text": a["message"]})
    else:
        payload = json.dumps({"text": a["message"]})
    qpayload = shlex.quote(payload)
    qurl = shlex.quote(url)
    return f"curl -X POST -H 'Content-Type: application/json' -d {qpayload} {qurl}"


def _bind_broadcast_message(a, get_channel_url):
    url = get_channel_url(a["channel"])
    payload = json.dumps({"text": a["message"]})
    qpayload = shlex.quote(payload)
    qurl = shlex.quote(url)
    return f"curl -X POST -H 'Content-Type: application/json' -d {qpayload} {qurl}"


def _bind_react_to_message(a, get_channel_url):
    # NOTE: preserves a pre-existing quirk -- the JSON payload values are not
    # quoted inside the braces. Reproduced verbatim for behavior parity.
    qchannel = shlex.quote(a["channel"])
    qts = shlex.quote(a["message_id"])
    qname = shlex.quote(a["emoji"])
    payload = f'{{"channel": {qchannel}, "timestamp": {qts}, "name": {qname}}}'
    return (
        "curl -X POST -H 'Authorization: Bearer xoxb-token' "
        "-H 'Content-Type: application/json' -d "
        f"'{payload}' https://slack.com/api/reactions.add"
    )


def _bind_search_grep(a, get_channel_url):
    path = a.get("path", ".")
    _, ext = os.path.splitext(path)
    pattern = shlex.quote(a["pattern"])
    qpath = shlex.quote(path)
    if ext:
        return f"grep -n {pattern} {qpath}"
    return f"grep -rn {pattern} {qpath}"


def _bind_run_script(a, get_channel_url):
    script_path = a["script_path"]
    args_list = a.get("args") or []
    joined = " ".join(shlex.quote(x) for x in args_list)
    qscript = shlex.quote(script_path)
    if script_path.endswith(".py"):
        return f"python3 {qscript} {joined}"
    if script_path.endswith(".sh"):
        return f"bash {qscript} {joined}"
    return DirectResult(
        [
            {
                "type": "text",
                "text": (
                    f"Error: Unsupported script file extension for {script_path}. "
                    "Must be .py or .sh."
                ),
            }
        ]
    )


def _bind_decompress_archive(a, get_channel_url):
    archive_path = a["archive_path"]
    out_dir = a.get("output_dir")
    qarchive = shlex.quote(archive_path)
    out_flag = f" -d {shlex.quote(out_dir)}" if out_dir else ""
    if archive_path.endswith(".zip"):
        password = a.get("password")
        pwd_flag = f" -P {shlex.quote(password)}" if password else ""
        return f"unzip{pwd_flag}{out_flag} {qarchive}"
    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        out_flag = f" -C {shlex.quote(out_dir)}" if out_dir else ""
        return f"tar -xzf {qarchive}{out_flag}"
    return DirectResult(
        [
            {
                "type": "text",
                "text": (
                    f"Error: Unsupported archive file format for {archive_path}. "
                    "Must be .zip, .tar.gz, or .tgz."
                ),
            }
        ]
    )


_COMPOSITE_BINDERS: dict[str, Callable[..., Any]] = {
    "write_file": _bind_write_file,
    "append_file": _bind_append_file,
    "send_message": _bind_send_message,
    "broadcast_message": _bind_broadcast_message,
    "react_to_message": _bind_react_to_message,
    "search_grep": _bind_search_grep,
    "run_script": _bind_run_script,
    "decompress_archive": _bind_decompress_archive,
}



# Projected-tool CLI bindings (see gen_mcp_facade.py for the tuple grammar).
_BINDINGS = {   'add_cron_job': (   'system cron add',
                        [   ('name', 'flag', '--name'),
                            ('cron_schedule', 'flag', '--cron'),
                            ('message', 'flag', '--message')]),
    'check_delegated_tasks_status': ('agent status', [('agent', 'flag', '--agent')]),
    'check_system_health': ('system health', []),
    'create_agent': ('agent add', [('name', 'pos'), ('model', 'flag', '--model')]),
    'create_calendar_event': (   'calendar add-event',
                                 [   ('title', 'flag', '--title'),
                                     ('start_time', 'flag', '--start'),
                                     ('location', 'flag', '--location'),
                                     (   'attendees',
                                         'array',
                                         '--attendees',
                                         {'sep': ','})]),
    'create_message_poll': (   'message poll',
                               [   ('channel', 'flag', '--channel'),
                                   ('question', 'flag', '--question'),
                                   ('options', 'array', '--options', {'sep': ','})]),
    'delegate_task': (   'agent delegate',
                         [   ('agent', 'flag', '--agent'),
                             ('instruction', 'flag', '--instruction')]),
    'delete_calendar_event': ('calendar delete-event', [('id', 'flag', '--id')]),
    'delete_file': ('rm -f', [('path', 'pos')]),
    'deploy_approve': (   'deploy approve',
                          [   ('change', 'flag', '--change'),
                              ('approver', 'flag', '--approver')]),
    'deploy_cutover': ('deploy cutover', [('change', 'flag', '--change')]),
    'deploy_execute': ('deploy execute', [('change', 'flag', '--change')]),
    'deploy_gate_set': (   'deploy gate-set',
                           [('gate', 'flag', '--gate'), ('state', 'flag', '--state')]),
    'deploy_monitor': ('deploy monitor', [('change', 'flag', '--change')]),
    'deploy_request': ('deploy request', [('change', 'flag', '--change')]),
    'deploy_rollback': ('deploy rollback', [('change', 'flag', '--change')]),
    'deploy_status': (   'deploy status',
                         [('change', 'flag', '--change', {'truthy': True})]),
    'describe_asset': ('asset describe', [('id', 'flag', '--id')]),
    'disable_plugin': ('plugin disable', [('name', 'pos')]),
    'enable_plugin': ('plugin enable', [('name', 'pos')]),
    'execute_db_query': ('sqlite3', [('db_path', 'pos'), ('query', 'pos')]),
    'get_asset_info': ('asset info', [('id', 'flag', '--id')]),
    'get_config_value': ('system config get', [('path', 'pos')]),
    'get_plugin_skill_info': ('plugin skill info', [('name', 'pos')]),
    'get_system_status': ('system status', [('json_format', 'bool', '--json')]),
    'get_today_date': ('calendar today', [('timezone', 'flag', '--timezone')]),
    'get_weather': ('weather get', [('location', 'flag', '--location')]),
    'get_weather_alerts': ('weather alerts', [('location', 'flag', '--location')]),
    'get_weather_forecast': (   'weather forecast',
                                [   ('location', 'flag', '--location'),
                                    ('days', 'flagint', '--days')]),
    'git_add': ('git add', [('path', 'pos')]),
    'git_commit': ('git commit', [('message', 'flag', '-m')]),
    'git_status': ('git status', []),
    'http_get': ('curl', [('url', 'pos')]),
    'install_plugin': ('plugin install', [('name', 'pos')]),
    'list_agents': ('agent list', []),
    'list_assets': ('asset list', [('kind', 'flag', '--kind')]),
    'list_calendar_events': (   'calendar list',
                                [   ('from_date', 'flag', '--from'),
                                    ('to_date', 'flag', '--to')]),
    'list_cron_jobs': ('system cron list', []),
    'list_devices': ('system devices list', []),
    'list_emails': ('email list', [('folder', 'flag', '--folder')]),
    'list_files': ('ls -la', [('path', 'pos', {'default': '.'})]),
    'list_plugins': ('plugin list', []),
    'lock_acquire': (   'lock acquire',
                        [   ('resource', 'flag', '--resource'),
                            ('section', 'flag', '--section', {'truthy': True}),
                            ('holder', 'flag', '--holder', {'truthy': True})]),
    'lock_list': ('lock list', []),
    'lock_release': (   'lock release',
                        [   ('resource', 'flag', '--resource'),
                            ('section', 'flag', '--section', {'truthy': True})]),
    'lock_status': (   'lock status',
                       [   ('resource', 'flag', '--resource'),
                           ('section', 'flag', '--section', {'truthy': True})]),
    'mark_email': ('email mark', [('id', 'flag', '--id'), ('flag', 'flag', '--flag')]),
    'move_email': (   'email move',
                      [('id', 'flag', '--id'), ('folder', 'flag', '--folder')]),
    'move_file': ('mv', [('src', 'pos'), ('dst', 'pos')]),
    'open_asset': ('asset open', [('id', 'flag', '--id')]),
    'pair_device': ('system devices pair', []),
    'read_asset_text': ('asset read-text', [('id', 'flag', '--id')]),
    'read_email': ('email read', [('id', 'flag', '--id')]),
    'read_file': ('cat', [('path', 'pos')]),
    'reply_to_email': (   'email reply',
                          [('id', 'flag', '--id'), ('body', 'flag', '--body')]),
    'run_jq_query': ('jq', [('filter', 'pos'), ('path', 'pos')]),
    'run_security_audit': ('system security audit', []),
    'run_system_diagnostics': ('system doctor', []),
    'run_system_setup': ('system setup', []),
    'search_emails': ('email search', [('query', 'flag', '--query')]),
    'search_messages': (   'message search',
                           [   ('channel', 'flag', '--channel'),
                               ('query', 'flag', '--query')]),
    'send_email': (   'email send',
                      [   ('to_address', 'flag', '--to'),
                          ('subject', 'flag', '--subject'),
                          ('body', 'flag', '--body')]),
    'set_agent_identity': (   'agent set-identity',
                              [('name', 'pos'), ('role', 'flag', '--role')]),
    'set_config_value': ('system config set', [('path', 'pos'), ('value', 'pos')]),
    'set_default_model': ('system models set', [('model', 'pos')]),
    'start_gateway': ('system gateway start', []),
    'undo': ('undo', [('steps', 'posint')]),
    'update_calendar_event': (   'calendar update-event',
                                 [   ('id', 'flag', '--id'),
                                     ('title', 'flag', '--title'),
                                     ('start_time', 'flag', '--start'),
                                     ('location', 'flag', '--location')]),
    'wait': ('wait', [('seconds', 'posint')])}

_COMPOSITE = {   'append_file',
    'broadcast_message',
    'decompress_archive',
    'react_to_message',
    'run_script',
    'search_grep',
    'send_message',
    'write_file'}

# Curated tool schemas (tools/list payload), verbatim, in definition order.
TOOL_SCHEMAS = [   {   'description': 'List virtual workspace files.',
        'inputSchema': {   'properties': {   'path': {   'description': 'Optional '
                                                                        'subdirectory '
                                                                        'path to list.',
                                                         'type': 'string'}},
                           'type': 'object'},
        'name': 'list_files'},
    {   'description': 'Create or overwrite a virtual workspace file with content.',
        'inputSchema': {   'properties': {   'content': {   'description': 'The file '
                                                                           'contents.',
                                                            'type': 'string'},
                                             'path': {   'description': 'Relative path '
                                                                        'to write to.',
                                                         'type': 'string'}},
                           'required': ['path', 'content'],
                           'type': 'object'},
        'name': 'write_file'},
    {   'description': 'Read the content of a virtual workspace file.',
        'inputSchema': {   'properties': {   'path': {   'description': 'Relative path '
                                                                        'to the file.',
                                                         'type': 'string'}},
                           'required': ['path'],
                           'type': 'object'},
        'name': 'read_file'},
    {   'description': 'Move or rename a virtual workspace file.',
        'inputSchema': {   'properties': {   'dst': {   'description': 'Destination '
                                                                       'path.',
                                                        'type': 'string'},
                                             'src': {   'description': 'Source path.',
                                                        'type': 'string'}},
                           'required': ['src', 'dst'],
                           'type': 'object'},
        'name': 'move_file'},
    {   'description': 'Append content to a virtual workspace file.',
        'inputSchema': {   'properties': {   'content': {   'description': 'Content to '
                                                                           'append.',
                                                            'type': 'string'},
                                             'path': {   'description': 'Relative path '
                                                                        'to the file.',
                                                         'type': 'string'}},
                           'required': ['path', 'content'],
                           'type': 'object'},
        'name': 'append_file'},
    {   'description': 'Delete a virtual workspace file.',
        'inputSchema': {   'properties': {   'path': {   'description': 'Relative path '
                                                                        'to the file '
                                                                        'to delete.',
                                                         'type': 'string'}},
                           'required': ['path'],
                           'type': 'object'},
        'name': 'delete_file'},
    {   'description': 'List visible multimodal assets.',
        'inputSchema': {   'properties': {   'kind': {   'description': 'Optional '
                                                                        'filter by '
                                                                        'asset kind.',
                                                         'type': 'string'}},
                           'type': 'object'},
        'name': 'list_assets'},
    {   'description': 'Show metadata for a visible asset.',
        'inputSchema': {   'properties': {   'id': {   'description': 'The unique '
                                                                      'asset ID.',
                                                       'type': 'string'}},
                           'required': ['id'],
                           'type': 'object'},
        'name': 'get_asset_info'},
    {   'description': 'Check for text attached to a visible asset. Assets here '
                       'carry no OCR or alt-text, so this reports none; use '
                       'open_asset to view the asset and read it directly.',
        'inputSchema': {   'properties': {   'id': {   'description': 'The unique '
                                                                      'asset ID.',
                                                       'type': 'string'}},
                           'required': ['id'],
                           'type': 'object'},
        'name': 'read_asset_text'},
    {   'description': 'Show a visible asset\'s identity and structure (kind, '
                       'path, MIME, dimensions). Its contents are visual and are '
                       'not described here; use open_asset to read them.',
        'inputSchema': {   'properties': {   'id': {   'description': 'The unique '
                                                                      'asset ID.',
                                                       'type': 'string'}},
                           'required': ['id'],
                           'type': 'object'},
        'name': 'describe_asset'},
    {   'description': 'Open an asset reference for multimodal inspection.',
        'inputSchema': {   'properties': {   'id': {   'description': 'The unique '
                                                                      'asset ID.',
                                                       'type': 'string'}},
                           'required': ['id'],
                           'type': 'object'},
        'name': 'open_asset'},
    {   'description': 'Send a message to a channel or another agent.',
        'inputSchema': {   'properties': {   'channel': {   'description': 'Chat '
                                                                           'channel '
                                                                           'name (e.g. '
                                                                           '#incident-war-room).',
                                                            'type': 'string'},
                                             'message': {   'description': 'The '
                                                                           'message '
                                                                           'text.',
                                                            'type': 'string'},
                                             'target': {   'description': 'Target '
                                                                          'agent '
                                                                          'name/LDAP.',
                                                           'type': 'string'}},
                           'required': ['channel', 'target', 'message'],
                           'type': 'object'},
        'name': 'send_message'},
    {   'description': 'Broadcast a message to a channel.',
        'inputSchema': {   'properties': {   'channel': {   'description': 'Chat '
                                                                           'channel '
                                                                           'name.',
                                                            'type': 'string'},
                                             'message': {   'description': 'The '
                                                                           'message '
                                                                           'text.',
                                                            'type': 'string'}},
                           'required': ['channel', 'message'],
                           'type': 'object'},
        'name': 'broadcast_message'},
    {   'description': 'Search messages within a channel.',
        'inputSchema': {   'properties': {   'channel': {   'description': 'Chat '
                                                                           'channel '
                                                                           'name.',
                                                            'type': 'string'},
                                             'query': {   'description': 'Search '
                                                                         'query.',
                                                          'type': 'string'}},
                           'required': ['channel', 'query'],
                           'type': 'object'},
        'name': 'search_messages'},
    {   'description': 'React to a message with an emoji.',
        'inputSchema': {   'properties': {   'channel': {   'description': 'Chat '
                                                                           'channel '
                                                                           'name.',
                                                            'type': 'string'},
                                             'emoji': {   'description': 'Emoji '
                                                                         'character.',
                                                          'type': 'string'},
                                             'message_id': {   'description': 'Target '
                                                                              'message '
                                                                              'ID.',
                                                               'type': 'string'}},
                           'required': ['channel', 'message_id', 'emoji'],
                           'type': 'object'},
        'name': 'react_to_message'},
    {   'description': 'Create a message poll in a channel.',
        'inputSchema': {   'properties': {   'channel': {   'description': 'Chat '
                                                                           'channel '
                                                                           'name.',
                                                            'type': 'string'},
                                             'options': {   'description': 'List of '
                                                                           'options.',
                                                            'items': {'type': 'string'},
                                                            'type': 'array'},
                                             'question': {   'description': 'Poll '
                                                                            'question '
                                                                            'text.',
                                                             'type': 'string'}},
                           'required': ['channel', 'question', 'options'],
                           'type': 'object'},
        'name': 'create_message_poll'},
    {   'description': 'Show system status.',
        'inputSchema': {   'properties': {   'json_format': {   'description': 'Whether '
                                                                               'to '
                                                                               'return '
                                                                               'status '
                                                                               'in '
                                                                               'JSON '
                                                                               'format.',
                                                                'type': 'boolean'}},
                           'type': 'object'},
        'name': 'get_system_status'},
    {   'description': 'Check gateway health.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'check_system_health'},
    {   'description': 'Run system diagnostics.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'run_system_diagnostics'},
    {   'description': 'Start the gateway.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'start_gateway'},
    {   'description': 'Set default model.',
        'inputSchema': {   'properties': {   'model': {   'description': 'Model '
                                                                         'identifier.',
                                                          'type': 'string'}},
                           'required': ['model'],
                           'type': 'object'},
        'name': 'set_default_model'},
    {   'description': 'Read config value.',
        'inputSchema': {   'properties': {   'path': {   'description': 'Exact config '
                                                                        'path (e.g. '
                                                                        'agent.model).',
                                                         'type': 'string'}},
                           'required': ['path'],
                           'type': 'object'},
        'name': 'get_config_value'},
    {   'description': 'Set config value.',
        'inputSchema': {   'properties': {   'path': {   'description': 'Exact config '
                                                                        'path.',
                                                         'type': 'string'},
                                             'value': {   'description': 'Value to '
                                                                         'set.',
                                                          'type': 'string'}},
                           'required': ['path', 'value'],
                           'type': 'object'},
        'name': 'set_config_value'},
    {   'description': 'Add a cron job.',
        'inputSchema': {   'properties': {   'cron_schedule': {   'description': 'Cron '
                                                                                 'expression '
                                                                                 'schedule.',
                                                                  'type': 'string'},
                                             'message': {   'description': 'The '
                                                                           'message to '
                                                                           'trigger.',
                                                            'type': 'string'},
                                             'name': {   'description': 'Unique cron '
                                                                        'job name.',
                                                         'type': 'string'}},
                           'required': ['name', 'cron_schedule', 'message'],
                           'type': 'object'},
        'name': 'add_cron_job'},
    {   'description': 'List cron jobs.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'list_cron_jobs'},
    {   'description': 'Run security audit.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'run_security_audit'},
    {   'description': 'Run system setup.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'run_system_setup'},
    {   'description': 'Pair a device.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'pair_device'},
    {   'description': 'List devices.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'list_devices'},
    {   'description': 'Wait for a specified number of virtual seconds.',
        'inputSchema': {   'properties': {   'seconds': {   'description': 'Seconds to '
                                                                           'wait.',
                                                            'type': 'integer'}},
                           'required': ['seconds'],
                           'type': 'object'},
        'name': 'wait'},
    {   'description': 'Undo the last N steps.',
        'inputSchema': {   'properties': {   'steps': {   'description': 'Steps to '
                                                                         'undo.',
                                                          'type': 'integer'}},
                           'type': 'object'},
        'name': 'undo'},
    {   'description': 'Create an agent.',
        'inputSchema': {   'properties': {   'model': {   'description': 'Optional '
                                                                         'model name.',
                                                          'type': 'string'},
                                             'name': {   'description': 'Agent name.',
                                                         'type': 'string'}},
                           'required': ['name'],
                           'type': 'object'},
        'name': 'create_agent'},
    {   'description': 'List agents.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'list_agents'},
    {   'description': 'Set agent identity.',
        'inputSchema': {   'properties': {   'name': {   'description': 'Agent name.',
                                                         'type': 'string'},
                                             'role': {   'description': 'Role to '
                                                                        'assign.',
                                                         'type': 'string'}},
                           'required': ['name', 'role'],
                           'type': 'object'},
        'name': 'set_agent_identity'},
    {   'description': 'Delegate a task to a subagent.',
        'inputSchema': {   'properties': {   'agent': {   'description': 'Recipient '
                                                                         'agent name.',
                                                          'type': 'string'},
                                             'instruction': {   'description': 'Task '
                                                                               'instruction '
                                                                               'text.',
                                                                'type': 'string'}},
                           'required': ['agent', 'instruction'],
                           'type': 'object'},
        'name': 'delegate_task'},
    {   'description': 'Check the status of pending delegated agent tasks.',
        'inputSchema': {   'properties': {   'agent': {   'description': 'Optional '
                                                                         'agent name '
                                                                         'filter.',
                                                          'type': 'string'}},
                           'type': 'object'},
        'name': 'check_delegated_tasks_status'},
    {   'description': 'List calendar events.',
        'inputSchema': {   'properties': {   'from_date': {   'description': 'Optional '
                                                                             'start '
                                                                             'date '
                                                                             'filter.',
                                                              'type': 'string'},
                                             'to_date': {   'description': 'Optional '
                                                                           'end date '
                                                                           'filter.',
                                                            'type': 'string'}},
                           'type': 'object'},
        'name': 'list_calendar_events'},
    {   'description': 'Create a calendar event.',
        'inputSchema': {   'properties': {   'attendees': {   'description': 'Optional '
                                                                             'attendees '
                                                                             'list.',
                                                              'items': {   'type': 'string'},
                                                              'type': 'array'},
                                             'location': {   'description': 'Optional '
                                                                            'location.',
                                                             'type': 'string'},
                                             'start_time': {   'description': 'Start '
                                                                              'datetime.',
                                                               'type': 'string'},
                                             'title': {   'description': 'Event title.',
                                                          'type': 'string'}},
                           'required': ['title', 'start_time'],
                           'type': 'object'},
        'name': 'create_calendar_event'},
    {   'description': 'Update a calendar event.',
        'inputSchema': {   'properties': {   'id': {   'description': 'Target event '
                                                                      'ID.',
                                                       'type': 'string'},
                                             'location': {   'description': 'New '
                                                                            'location.',
                                                             'type': 'string'},
                                             'start_time': {   'description': 'New '
                                                                              'start '
                                                                              'datetime.',
                                                               'type': 'string'},
                                             'title': {   'description': 'New event '
                                                                         'title.',
                                                          'type': 'string'}},
                           'required': ['id'],
                           'type': 'object'},
        'name': 'update_calendar_event'},
    {   'description': 'Delete a calendar event.',
        'inputSchema': {   'properties': {   'id': {   'description': 'Target event '
                                                                      'ID.',
                                                       'type': 'string'}},
                           'required': ['id'],
                           'type': 'object'},
        'name': 'delete_calendar_event'},
    {   'description': 'Return local date for a timezone.',
        'inputSchema': {   'properties': {   'timezone': {   'description': 'Target '
                                                                            'timezone.',
                                                             'type': 'string'}},
                           'required': ['timezone'],
                           'type': 'object'},
        'name': 'get_today_date'},
    {   'description': 'List emails.',
        'inputSchema': {   'properties': {   'folder': {   'description': 'Optional '
                                                                          'folder '
                                                                          'filter.',
                                                           'type': 'string'}},
                           'type': 'object'},
        'name': 'list_emails'},
    {   'description': 'Read an email.',
        'inputSchema': {   'properties': {   'id': {   'description': 'Email ID.',
                                                       'type': 'string'}},
                           'required': ['id'],
                           'type': 'object'},
        'name': 'read_email'},
    {   'description': 'Send an email.',
        'inputSchema': {   'properties': {   'body': {   'description': 'Email body.',
                                                         'type': 'string'},
                                             'subject': {   'description': 'Email '
                                                                           'subject.',
                                                            'type': 'string'},
                                             'to_address': {   'description': 'Recipient '
                                                                              'email '
                                                                              'address.',
                                                               'type': 'string'}},
                           'required': ['to_address', 'subject', 'body'],
                           'type': 'object'},
        'name': 'send_email'},
    {   'description': 'Reply to an email.',
        'inputSchema': {   'properties': {   'body': {   'description': 'Reply body.',
                                                         'type': 'string'},
                                             'id': {   'description': 'Target email '
                                                                      'ID.',
                                                       'type': 'string'}},
                           'required': ['id', 'body'],
                           'type': 'object'},
        'name': 'reply_to_email'},
    {   'description': 'Move an email to a folder.',
        'inputSchema': {   'properties': {   'folder': {   'description': 'Destination '
                                                                          'folder.',
                                                           'type': 'string'},
                                             'id': {   'description': 'Email ID.',
                                                       'type': 'string'}},
                           'required': ['id', 'folder'],
                           'type': 'object'},
        'name': 'move_email'},
    {   'description': 'Mark an email read or unread.',
        'inputSchema': {   'properties': {   'flag': {   'description': 'State flag.',
                                                         'enum': ['read', 'unread'],
                                                         'type': 'string'},
                                             'id': {   'description': 'Email ID.',
                                                       'type': 'string'}},
                           'required': ['id', 'flag'],
                           'type': 'object'},
        'name': 'mark_email'},
    {   'description': 'Search emails by query.',
        'inputSchema': {   'properties': {   'query': {   'description': 'Search '
                                                                         'query.',
                                                          'type': 'string'}},
                           'required': ['query'],
                           'type': 'object'},
        'name': 'search_emails'},
    {   'description': 'Get current weather.',
        'inputSchema': {   'properties': {   'location': {   'description': 'Location '
                                                                            'name.',
                                                             'type': 'string'}},
                           'required': ['location'],
                           'type': 'object'},
        'name': 'get_weather'},
    {   'description': 'Get weather forecast.',
        'inputSchema': {   'properties': {   'days': {   'description': 'Forecast '
                                                                        'days.',
                                                         'type': 'integer'},
                                             'location': {   'description': 'Location '
                                                                            'name.',
                                                             'type': 'string'}},
                           'required': ['location', 'days'],
                           'type': 'object'},
        'name': 'get_weather_forecast'},
    {   'description': 'Get weather alerts.',
        'inputSchema': {   'properties': {   'location': {   'description': 'Location '
                                                                            'name.',
                                                             'type': 'string'}},
                           'required': ['location'],
                           'type': 'object'},
        'name': 'get_weather_alerts'},
    {   'description': 'Fetch a URL through the HTTP backend.',
        'inputSchema': {   'properties': {   'url': {   'description': 'Target URL.',
                                                        'type': 'string'}},
                           'required': ['url'],
                           'type': 'object'},
        'name': 'http_get'},
    {   'description': 'Enable a plugin.',
        'inputSchema': {   'properties': {   'name': {   'description': 'Plugin name.',
                                                         'type': 'string'}},
                           'required': ['name'],
                           'type': 'object'},
        'name': 'enable_plugin'},
    {   'description': 'Disable a plugin.',
        'inputSchema': {   'properties': {   'name': {   'description': 'Plugin name.',
                                                         'type': 'string'}},
                           'required': ['name'],
                           'type': 'object'},
        'name': 'disable_plugin'},
    {   'description': 'List plugins.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'list_plugins'},
    {   'description': 'Install a plugin.',
        'inputSchema': {   'properties': {   'name': {   'description': 'Plugin name.',
                                                         'type': 'string'}},
                           'required': ['name'],
                           'type': 'object'},
        'name': 'install_plugin'},
    {   'description': 'Show skill info.',
        'inputSchema': {   'properties': {   'name': {   'description': 'Plugin/skill '
                                                                        'name.',
                                                         'type': 'string'}},
                           'required': ['name'],
                           'type': 'object'},
        'name': 'get_plugin_skill_info'},
    {   'description': 'Show the working tree status.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'git_status'},
    {   'description': 'Add file contents to the staging area.',
        'inputSchema': {   'properties': {   'path': {   'description': 'Path to file '
                                                                        'to stage.',
                                                         'type': 'string'}},
                           'required': ['path'],
                           'type': 'object'},
        'name': 'git_add'},
    {   'description': 'Record changes to the repository.',
        'inputSchema': {   'properties': {   'message': {   'description': 'Commit '
                                                                           'message.',
                                                            'type': 'string'}},
                           'required': ['message'],
                           'type': 'object'},
        'name': 'git_commit'},
    {   'description': 'Query JSON data using jq filter syntax.',
        'inputSchema': {   'properties': {   'filter': {   'description': 'The jq '
                                                                          'query '
                                                                          'filter '
                                                                          '(e.g. '
                                                                          "'.items[0].id').",
                                                           'type': 'string'},
                                             'path': {   'description': 'Relative path '
                                                                        'to the target '
                                                                        'JSON file.',
                                                         'type': 'string'}},
                           'required': ['filter', 'path'],
                           'type': 'object'},
        'name': 'run_jq_query'},
    {   'description': 'Search for patterns in files recursively.',
        'inputSchema': {   'properties': {   'path': {   'description': 'Target file '
                                                                        'or directory '
                                                                        'to search. '
                                                                        'Defaults to '
                                                                        "'.'.",
                                                         'type': 'string'},
                                             'pattern': {   'description': 'Regular '
                                                                           'expression '
                                                                           'pattern to '
                                                                           'find.',
                                                            'type': 'string'}},
                           'required': ['pattern'],
                           'type': 'object'},
        'name': 'search_grep'},
    {   'description': 'Execute a local python (.py) or bash (.sh) script in the '
                       'workspace.',
        'inputSchema': {   'properties': {   'args': {   'description': 'Optional list '
                                                                        'of arguments '
                                                                        'to pass to '
                                                                        'the script.',
                                                         'items': {'type': 'string'},
                                                         'type': 'array'},
                                             'script_path': {   'description': 'Relative '
                                                                               'path '
                                                                               'to the '
                                                                               'script '
                                                                               'file.',
                                                                'type': 'string'}},
                           'required': ['script_path'],
                           'type': 'object'},
        'name': 'run_script'},
    {   'description': 'Decompress a zip (.zip) or tar (.tar.gz) archive.',
        'inputSchema': {   'properties': {   'archive_path': {   'description': 'Relative '
                                                                                'path '
                                                                                'to '
                                                                                'the '
                                                                                'archive '
                                                                                'file.',
                                                                 'type': 'string'},
                                             'output_dir': {   'description': 'Optional '
                                                                              'relative '
                                                                              'output '
                                                                              'directory.',
                                                               'type': 'string'},
                                             'password': {   'description': 'Optional '
                                                                            'decryption '
                                                                            'password '
                                                                            'for zip '
                                                                            'files.',
                                                             'type': 'string'}},
                           'required': ['archive_path'],
                           'type': 'object'},
        'name': 'decompress_archive'},
    {   'description': 'Execute an SQL query on an SQLite (.sqlite) database file.',
        'inputSchema': {   'properties': {   'db_path': {   'description': 'Relative '
                                                                           'path to '
                                                                           'the SQLite '
                                                                           'database '
                                                                           'file.',
                                                            'type': 'string'},
                                             'query': {   'description': 'The SQL '
                                                                         'query to '
                                                                         'execute.',
                                                          'type': 'string'}},
                           'required': ['db_path', 'query'],
                           'type': 'object'},
        'name': 'execute_db_query'},
    {   'description': 'Acquire an advisory lock over a named resource. Fails if the '
                       'resource is already held by a different holder.',
        'inputSchema': {   'properties': {   'holder': {   'description': 'Optional '
                                                                          'lock holder '
                                                                          'identity '
                                                                          '(defaults '
                                                                          'to the '
                                                                          'current '
                                                                          'actor).',
                                                           'type': 'string'},
                                             'resource': {   'description': 'The '
                                                                            'resource '
                                                                            'name to '
                                                                            'lock.',
                                                             'type': 'string'},
                                             'section': {   'description': 'Optional '
                                                                           'sub-section '
                                                                           'of the '
                                                                           'resource.',
                                                            'type': 'string'}},
                           'required': ['resource'],
                           'type': 'object'},
        'name': 'lock_acquire'},
    {   'description': 'Release an advisory lock over a named resource.',
        'inputSchema': {   'properties': {   'resource': {   'description': 'The '
                                                                            'resource '
                                                                            'name to '
                                                                            'unlock.',
                                                             'type': 'string'},
                                             'section': {   'description': 'Optional '
                                                                           'sub-section '
                                                                           'of the '
                                                                           'resource.',
                                                            'type': 'string'}},
                           'required': ['resource'],
                           'type': 'object'},
        'name': 'lock_release'},
    {   'description': 'List currently held locks.',
        'inputSchema': {'properties': {}, 'type': 'object'},
        'name': 'lock_list'},
    {   'description': 'Report whether a resource is free or held, and by whom.',
        'inputSchema': {   'properties': {   'resource': {   'description': 'The '
                                                                            'resource '
                                                                            'name to '
                                                                            'inspect.',
                                                             'type': 'string'},
                                             'section': {   'description': 'Optional '
                                                                           'sub-section '
                                                                           'of the '
                                                                           'resource.',
                                                            'type': 'string'}},
                           'required': ['resource'],
                           'type': 'object'},
        'name': 'lock_status'},
    {   'description': 'Register a deployment request for a change.',
        'inputSchema': {   'properties': {   'change': {   'description': 'The change '
                                                                          'identifier '
                                                                          'to deploy.',
                                                           'type': 'string'}},
                           'required': ['change'],
                           'type': 'object'},
        'name': 'deploy_request'},
    {   'description': 'Record an approval for a deployment change.',
        'inputSchema': {   'properties': {   'approver': {   'description': 'The '
                                                                            'approver '
                                                                            'identity.',
                                                             'type': 'string'},
                                             'change': {   'description': 'The change '
                                                                          'identifier.',
                                                           'type': 'string'}},
                           'required': ['change', 'approver'],
                           'type': 'object'},
        'name': 'deploy_approve'},
    {   'description': 'Create or update a readiness gate state '
                       '(ready|pending|failed|stale|expired).',
        'inputSchema': {   'properties': {   'gate': {   'description': 'The gate '
                                                                        'name.',
                                                         'type': 'string'},
                                             'state': {   'description': 'One of: '
                                                                         'ready, '
                                                                         'pending, '
                                                                         'failed, '
                                                                         'stale, '
                                                                         'expired.',
                                                          'type': 'string'}},
                           'required': ['gate', 'state'],
                           'type': 'object'},
        'name': 'deploy_gate_set'},
    {   'description': 'Execute a previously requested deployment.',
        'inputSchema': {   'properties': {   'change': {   'description': 'The change '
                                                                          'identifier '
                                                                          'to execute.',
                                                           'type': 'string'}},
                           'required': ['change'],
                           'type': 'object'},
        'name': 'deploy_execute'},
    {   'description': 'Cut over a deployment. Blocked unless every readiness gate is '
                       'ready.',
        'inputSchema': {   'properties': {   'change': {   'description': 'The change '
                                                                          'identifier '
                                                                          'to cut '
                                                                          'over.',
                                                           'type': 'string'}},
                           'required': ['change'],
                           'type': 'object'},
        'name': 'deploy_cutover'},
    {   'description': 'Roll back a deployment change.',
        'inputSchema': {   'properties': {   'change': {   'description': 'The change '
                                                                          'identifier '
                                                                          'to roll '
                                                                          'back.',
                                                           'type': 'string'}},
                           'required': ['change'],
                           'type': 'object'},
        'name': 'deploy_rollback'},
    {   'description': 'Report the current status of a deployment change.',
        'inputSchema': {   'properties': {   'change': {   'description': 'The change '
                                                                          'identifier '
                                                                          'to monitor.',
                                                           'type': 'string'}},
                           'required': ['change'],
                           'type': 'object'},
        'name': 'deploy_monitor'},
    {   'description': 'List readiness gates and deployment statuses.',
        'inputSchema': {   'properties': {   'change': {   'description': 'Optional '
                                                                          'change '
                                                                          'identifier '
                                                                          'to filter '
                                                                          'deployments.',
                                                           'type': 'string'}},
                           'type': 'object'},
        'name': 'deploy_status'}]

# --------------------------------------------------------------------------- #
# M3 append block: booking (reversible_action_hold) + pairing
# (multi_device_pairing) tools. Appended here -- rather than merged into the
# pprint literals above -- so the downstream _SCHEMA_BY_NAME / ALL_TOOL_NAMES /
# build_tools_list wiring picks them up automatically. The parity tests match
# tools by name (order-independent), so appending is contract-equivalent.
# --------------------------------------------------------------------------- #
_BINDINGS.update(
    {
        "booking_search": (
            "booking search",
            [
                ("resource", "flag", "--resource", {"truthy": True}),
                ("query", "flag", "--query", {"truthy": True}),
            ],
        ),
        "booking_hold": (
            "booking hold",
            [
                ("resource", "flag", "--resource"),
                ("holder", "flag", "--holder", {"truthy": True}),
                ("ttl", "flagint", "--ttl"),
            ],
        ),
        "booking_confirm": (
            "booking confirm",
            [
                ("resource", "flag", "--resource"),
                ("holder", "flag", "--holder", {"truthy": True}),
            ],
        ),
        "booking_release": (
            "booking release",
            [("resource", "flag", "--resource")],
        ),
        "booking_extend": (
            "booking extend",
            [
                ("resource", "flag", "--resource"),
                ("ttl", "flagint", "--ttl"),
            ],
        ),
        "booking_status": (
            "booking status",
            [("resource", "flag", "--resource", {"truthy": True})],
        ),
        "pairing_start": (
            "pairing start",
            [
                ("device", "flag", "--device"),
                ("ttl", "flagint", "--ttl"),
            ],
        ),
        "pairing_show_code": (
            "pairing show-code",
            [
                ("device", "flag", "--device"),
                ("code", "flag", "--code", {"truthy": True}),
            ],
        ),
        "pairing_enter_code": (
            "pairing enter-code",
            [
                ("device", "flag", "--device"),
                ("code", "flag", "--code", {"truthy": True}),
            ],
        ),
        "pairing_confirm": (
            "pairing confirm",
            [("device", "flag", "--device")],
        ),
        "pairing_status": (
            "pairing status",
            [("device", "flag", "--device", {"truthy": True})],
        ),
    }
)

TOOL_SCHEMAS += [
    {
        "description": "List resource holds and their availability (read-only).",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "Optional resource name to filter by.",
                    "type": "string",
                },
                "query": {
                    "description": "Optional substring to match against held "
                    "resource names.",
                    "type": "string",
                },
            },
            "type": "object",
        },
        "name": "booking_search",
    },
    {
        "description": "Place a reversible, time-limited hold on a resource.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource name to hold.",
                    "type": "string",
                },
                "holder": {
                    "description": "Optional hold owner identity (defaults to "
                    "the current actor).",
                    "type": "string",
                },
                "ttl": {
                    "description": "Optional hold lifetime in virtual seconds "
                    "(defaults to 300).",
                    "type": "integer",
                },
            },
            "required": ["resource"],
            "type": "object",
        },
        "name": "booking_hold",
    },
    {
        "description": "Confirm a hold before its window closes (fails if the "
        "hold has expired).",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource name whose hold to confirm.",
                    "type": "string",
                },
                "holder": {
                    "description": "Optional expected holder; confirmation "
                    "fails if it does not match.",
                    "type": "string",
                },
            },
            "required": ["resource"],
            "type": "object",
        },
        "name": "booking_confirm",
    },
    {
        "description": "Release (reverse) a hold on a resource.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource name to release.",
                    "type": "string",
                }
            },
            "required": ["resource"],
            "type": "object",
        },
        "name": "booking_release",
    },
    {
        "description": "Refresh a hold's time window before it expires.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource name whose hold to extend.",
                    "type": "string",
                },
                "ttl": {
                    "description": "Optional new lifetime in virtual seconds "
                    "(defaults to the hold's current ttl).",
                    "type": "integer",
                },
            },
            "required": ["resource"],
            "type": "object",
        },
        "name": "booking_extend",
    },
    {
        "description": "Show the status of resource holds (read-only).",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "Optional resource name to inspect.",
                    "type": "string",
                }
            },
            "type": "object",
        },
        "name": "booking_status",
    },
    {
        "description": "Open a time-bounded pairing session for a device.",
        "inputSchema": {
            "properties": {
                "device": {
                    "description": "The device identifier to pair.",
                    "type": "string",
                },
                "ttl": {
                    "description": "Optional session lifetime in virtual "
                    "seconds (defaults to 300).",
                    "type": "integer",
                },
            },
            "required": ["device"],
            "type": "object",
        },
        "name": "pairing_start",
    },
    {
        "description": "Reveal the pairing code for an open session.",
        "inputSchema": {
            "properties": {
                "device": {
                    "description": "The device identifier.",
                    "type": "string",
                },
                "code": {
                    "description": "Optional explicit code to use (defaults to "
                    "a derived code).",
                    "type": "string",
                },
            },
            "required": ["device"],
            "type": "object",
        },
        "name": "pairing_show_code",
    },
    {
        "description": "Enter the pairing code from the peer device.",
        "inputSchema": {
            "properties": {
                "device": {
                    "description": "The device identifier.",
                    "type": "string",
                },
                "code": {
                    "description": "Optional code to enter; must match the "
                    "shown code if provided.",
                    "type": "string",
                },
            },
            "required": ["device"],
            "type": "object",
        },
        "name": "pairing_enter_code",
    },
    {
        "description": "Complete the pairing once the code is entered (fails if "
        "the session has expired).",
        "inputSchema": {
            "properties": {
                "device": {
                    "description": "The device identifier to confirm.",
                    "type": "string",
                }
            },
            "required": ["device"],
            "type": "object",
        },
        "name": "pairing_confirm",
    },
    {
        "description": "Show the status of pairing sessions (read-only).",
        "inputSchema": {
            "properties": {
                "device": {
                    "description": "Optional device identifier to inspect.",
                    "type": "string",
                }
            },
            "type": "object",
        },
        "name": "pairing_status",
    },
]

# --------------------------------------------------------------------------- #
# M4 append block: approval (multi_party_access_permit / deadlock_multi_approval),
# containment (coordinated_containment), and review (non_overlapping_assignment)
# tools. Appended (not merged into the pprint literals above) for the same reason
# as the M3 block: the downstream _SCHEMA_BY_NAME / ALL_TOOL_NAMES /
# build_tools_list wiring picks them up automatically, and the parity tests match
# tools by name (order-independent), so appending is contract-equivalent.
# --------------------------------------------------------------------------- #
_BINDINGS.update(
    {
        "approval_request": (
            "approval request",
            [
                ("resource", "flag", "--resource"),
                ("scope", "flag", "--scope", {"truthy": True}),
                ("approvers", "flag", "--approvers", {"truthy": True}),
            ],
        ),
        "approval_approve": (
            "approval approve",
            [
                ("resource", "flag", "--resource"),
                ("approver", "flag", "--approver"),
                ("scope", "flag", "--scope", {"truthy": True}),
                ("ttl", "flagint", "--ttl"),
            ],
        ),
        "approval_grant": (
            "approval grant",
            [("resource", "flag", "--resource")],
        ),
        "approval_revoke": (
            "approval revoke",
            [
                ("resource", "flag", "--resource"),
                ("approver", "flag", "--approver", {"truthy": True}),
            ],
        ),
        "approval_provisional": (
            "approval provisional",
            [
                ("resource", "flag", "--resource"),
                ("approver", "flag", "--approver"),
                ("scope", "flag", "--scope", {"truthy": True}),
                ("ttl", "flagint", "--ttl"),
            ],
        ),
        "approval_escalate": (
            "approval escalate",
            [
                ("resource", "flag", "--resource"),
                ("to", "flag", "--to"),
            ],
        ),
        "approval_status": (
            "approval status",
            [("resource", "flag", "--resource", {"truthy": True})],
        ),
        "containment_revoke_sessions": (
            "containment revoke-sessions",
            [("account", "flag", "--account")],
        ),
        "containment_disable_forwarding": (
            "containment disable-forwarding",
            [("account", "flag", "--account")],
        ),
        "containment_isolate_device": (
            "containment isolate-device",
            [("device", "flag", "--device")],
        ),
        "containment_block_token": (
            "containment block-token",
            [("token", "flag", "--token")],
        ),
        "containment_status": (
            "containment status",
            [],
        ),
        "review_assign": (
            "review assign",
            [
                ("aspect", "flag", "--aspect"),
                ("reviewer", "flag", "--reviewer"),
            ],
        ),
        "review_submit": (
            "review submit",
            [
                ("aspect", "flag", "--aspect"),
                ("status", "flag", "--status"),
                ("issue", "flag", "--issue", {"truthy": True}),
            ],
        ),
        "review_sign": (
            "review sign",
            [("document", "flag", "--document")],
        ),
        "review_status": (
            "review status",
            [("document", "flag", "--document", {"truthy": True})],
        ),
    }
)

TOOL_SCHEMAS += [
    {
        "description": "Open an access request naming the required approvers and "
        "requested scope.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource to request access to.",
                    "type": "string",
                },
                "scope": {
                    "description": "Optional requested scope (e.g. read, write, "
                    "admin).",
                    "type": "string",
                },
                "approvers": {
                    "description": "Optional comma-separated list of required "
                    "approver roles.",
                    "type": "string",
                },
            },
            "required": ["resource"],
            "type": "object",
        },
        "name": "approval_request",
    },
    {
        "description": "Record one approver's approval, optionally scoped and "
        "time-limited.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource being approved.",
                    "type": "string",
                },
                "approver": {
                    "description": "The approver role recording the approval.",
                    "type": "string",
                },
                "scope": {
                    "description": "Optional scope this approver grants.",
                    "type": "string",
                },
                "ttl": {
                    "description": "Optional approval lifetime in virtual "
                    "seconds.",
                    "type": "integer",
                },
            },
            "required": ["resource", "approver"],
            "type": "object",
        },
        "name": "approval_approve",
    },
    {
        "description": "Grant access once every required approval is valid, "
        "unexpired, and scope-reconciled; fails otherwise.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource to grant access to.",
                    "type": "string",
                }
            },
            "required": ["resource"],
            "type": "object",
        },
        "name": "approval_grant",
    },
    {
        "description": "Revoke a resource's request, or one approver's approval.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource whose request/approval to "
                    "revoke.",
                    "type": "string",
                },
                "approver": {
                    "description": "Optional approver role; revoke only this "
                    "approver's approval.",
                    "type": "string",
                },
            },
            "required": ["resource"],
            "type": "object",
        },
        "name": "approval_revoke",
    },
    {
        "description": "Grant a provisional (deadlock-breaking) approval for an "
        "approver.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource being provisionally approved.",
                    "type": "string",
                },
                "approver": {
                    "description": "The approver role recording the provisional "
                    "approval.",
                    "type": "string",
                },
                "scope": {
                    "description": "Optional scope this provisional approval "
                    "grants.",
                    "type": "string",
                },
                "ttl": {
                    "description": "Optional provisional approval lifetime in "
                    "virtual seconds.",
                    "type": "integer",
                },
            },
            "required": ["resource", "approver"],
            "type": "object",
        },
        "name": "approval_provisional",
    },
    {
        "description": "Escalate a stuck approval to a higher authority to break "
        "a circular dependency.",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "The resource whose approval is stuck.",
                    "type": "string",
                },
                "to": {
                    "description": "The higher-authority role to escalate to.",
                    "type": "string",
                },
            },
            "required": ["resource", "to"],
            "type": "object",
        },
        "name": "approval_escalate",
    },
    {
        "description": "Show requests and approvals (read-only).",
        "inputSchema": {
            "properties": {
                "resource": {
                    "description": "Optional resource to filter by.",
                    "type": "string",
                }
            },
            "type": "object",
        },
        "name": "approval_status",
    },
    {
        "description": "Revoke all active sessions for the compromised account "
        "(one of four control points).",
        "inputSchema": {
            "properties": {
                "account": {
                    "description": "The compromised account whose sessions to "
                    "revoke.",
                    "type": "string",
                }
            },
            "required": ["account"],
            "type": "object",
        },
        "name": "containment_revoke_sessions",
    },
    {
        "description": "Disable mail/message forwarding rules on the account "
        "(one of four control points).",
        "inputSchema": {
            "properties": {
                "account": {
                    "description": "The account whose forwarding rules to "
                    "disable.",
                    "type": "string",
                }
            },
            "required": ["account"],
            "type": "object",
        },
        "name": "containment_disable_forwarding",
    },
    {
        "description": "Isolate the compromised device from the network (one of "
        "four control points).",
        "inputSchema": {
            "properties": {
                "device": {
                    "description": "The compromised device to isolate.",
                    "type": "string",
                }
            },
            "required": ["device"],
            "type": "object",
        },
        "name": "containment_isolate_device",
    },
    {
        "description": "Revoke and block the leaked access token (one of four "
        "control points).",
        "inputSchema": {
            "properties": {
                "token": {
                    "description": "The leaked access token to block.",
                    "type": "string",
                }
            },
            "required": ["token"],
            "type": "object",
        },
        "name": "containment_block_token",
    },
    {
        "description": "Show which control points are contained (read-only).",
        "inputSchema": {
            "properties": {},
            "type": "object",
        },
        "name": "containment_status",
    },
    {
        "description": "Assign a review aspect (legal, privacy, security, ...) "
        "to a reviewer.",
        "inputSchema": {
            "properties": {
                "aspect": {
                    "description": "The review aspect to assign.",
                    "type": "string",
                },
                "reviewer": {
                    "description": "The reviewer role that owns the aspect.",
                    "type": "string",
                },
            },
            "required": ["aspect", "reviewer"],
            "type": "object",
        },
        "name": "review_assign",
    },
    {
        "description": "Submit one aspect's verdict; a block carries its "
        "blocking issue.",
        "inputSchema": {
            "properties": {
                "aspect": {
                    "description": "The review aspect being submitted.",
                    "type": "string",
                },
                "status": {
                    "description": "The verdict: ok or blocked.",
                    "type": "string",
                },
                "issue": {
                    "description": "Optional blocking issue when status is "
                    "blocked.",
                    "type": "string",
                },
            },
            "required": ["aspect", "status"],
            "type": "object",
        },
        "name": "review_submit",
    },
    {
        "description": "Sign the document; blocked unless every aspect is ok "
        "with no outstanding blockers.",
        "inputSchema": {
            "properties": {
                "document": {
                    "description": "The document to sign.",
                    "type": "string",
                }
            },
            "required": ["document"],
            "type": "object",
        },
        "name": "review_sign",
    },
    {
        "description": "Show aspects, verdicts, and sign state (read-only).",
        "inputSchema": {
            "properties": {
                "document": {
                    "description": "Optional document to filter by.",
                    "type": "string",
                }
            },
            "type": "object",
        },
        "name": "review_status",
    },
]

# name -> {"description", "inputSchema"} for O(1) schema lookup.
_SCHEMA_BY_NAME = {t["name"]: t for t in TOOL_SCHEMAS}

# All curated tool names (projected + composite), for validation.
ALL_TOOL_NAMES = frozenset(t["name"] for t in TOOL_SCHEMAS)


def build_tools_list() -> list[dict[str, Any]]:
    """Return the curated ``tools/list`` payload (definition order)."""
    return TOOL_SCHEMAS


def _render_arg(binding: tuple, arguments: dict[str, Any], required: set) -> list[str]:
    name = binding[0]
    kind = binding[1]
    token = binding[2] if len(binding) >= 3 and isinstance(binding[2], str) else None
    opts = binding[-1] if isinstance(binding[-1], dict) else {}

    if kind == "bool":
        return [token] if arguments.get(name) else []

    is_required = name in required
    if name in arguments:
        val = arguments[name]
    elif "default" in opts:
        val = opts["default"]
    elif is_required:
        val = arguments[name]  # raises KeyError, matching legacy behavior
    else:
        return []

    if (not is_required) and opts.get("truthy") and not val:
        return []

    if kind == "pos":
        return [shlex.quote(str(val))]
    if kind == "posint":
        return [str(int(val))]
    if kind == "flag":
        return [token, shlex.quote(str(val))]
    if kind == "flagint":
        return [token, str(int(val))]
    if kind == "array":
        return [token, shlex.quote(opts.get("sep", ",").join(val))]
    raise ValueError(f"unknown arg bind kind: {kind!r}")


def build_cli(
    name: str,
    arguments: dict[str, Any],
    get_channel_url: Callable[[str], str] | None = None,
):
    """Build the workspace CLI string (or a :class:`DirectResult`) for a tool.

    Mirrors the legacy ``call_tool`` dispatch exactly. Raises ``ValueError`` for
    an unknown tool name.
    """
    if name in _COMPOSITE_BINDERS:
        return _COMPOSITE_BINDERS[name](arguments, get_channel_url)

    binding = _BINDINGS.get(name)
    if binding is None:
        raise ValueError(f"Unknown MCP tool: {name}")

    command, arg_specs = binding
    schema = _SCHEMA_BY_NAME[name]["inputSchema"]
    required = set(schema.get("required", []))
    parts = [command]
    for spec in arg_specs:
        parts.extend(_render_arg(spec, arguments, required))
    return " ".join(parts)

