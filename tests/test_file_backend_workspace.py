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

from clasharness.backend.file_backend import FileSystemBackend


def test_file_backend_loads_state_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "evidence.md").write_text("Final evidence token", encoding="utf-8")
    (workspace / "nested").mkdir()
    (workspace / "nested" / "session.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")

    backend = FileSystemBackend()
    backend.initialize(str(tmp_path), {"OPENCLAW_STATE_DIR": str(tmp_path)})

    assert "evidence.md" in backend.execute_cli("file list --path /").stdout
    read = backend.execute_cli("file read --path /evidence.md")
    assert read.exit_code == 0
    assert "Final evidence token" in read.stdout
    nested = backend.execute_cli("file read --path /nested/session.jsonl")
    assert nested.exit_code == 0
    assert "hello" in nested.stdout


def test_file_backend_update_alias(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test.txt").write_text("initial content", encoding="utf-8")

    backend = FileSystemBackend()
    backend.initialize(str(tmp_path), {"OPENCLAW_STATE_DIR": str(tmp_path)})

    res = backend.execute_cli("file update --path /test.txt --content 'updated content'")
    assert res.exit_code == 0
    read = backend.execute_cli("file read --path /test.txt")
    assert "updated content" in read.stdout


def test_file_backend_append_emits_gradeable_effect(tmp_path) -> None:
    # I4: `file append` must emit a first-class ``files_appended`` effect so that
    # content appended to a report is visible to the effect stream (previously an
    # append produced no state_changes and scored zero on content checks).
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "report.md").write_text("Findings: ", encoding="utf-8")

    backend = FileSystemBackend()
    backend.initialize(str(tmp_path), {"OPENCLAW_STATE_DIR": str(tmp_path)})

    res = backend.execute_cli("file append --path /report.md --content 'root cause is v4.2'")
    assert res.exit_code == 0

    appended = res.state_changes["files_appended"]
    assert len(appended) == 1
    assert appended[0]["path"] == "/report.md"
    assert appended[0]["content"] == "root cause is v4.2"
    # ``full_content`` is the authoritative post-append body (seed + appended),
    # which the environment folds into files_created for content checkers.
    assert appended[0]["full_content"] == "Findings: root cause is v4.2"

    # And the append is persisted to disk / the virtual FS.
    read = backend.execute_cli("file read --path /report.md")
    assert "root cause is v4.2" in read.stdout


def test_file_backend_append_missing_file_errors(tmp_path) -> None:
    # Appending to a non-existent file stays an error and emits no effect.
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    backend = FileSystemBackend()
    backend.initialize(str(tmp_path), {"OPENCLAW_STATE_DIR": str(tmp_path)})

    res = backend.execute_cli("file append --path /nope.md --content data123")
    assert res.exit_code == 1
    # Error path leaves state_changes unset (None); no effect is emitted.
    assert not (res.state_changes or {}).get("files_appended")


def _fresh_backend(tmp_path) -> FileSystemBackend:
    (tmp_path / "workspace").mkdir(exist_ok=True)
    backend = FileSystemBackend()
    backend.initialize(str(tmp_path), {"OPENCLAW_STATE_DIR": str(tmp_path)})
    return backend


def test_file_create_writes_full_multiline_content(tmp_path) -> None:
    # Regression: a real multi-line ``--content`` report body (genuine newlines,
    # closing quote on a later line) must be written in full. Peeling the content
    # before tokenizing keeps shlex from choking on the embedded newlines.
    backend = _fresh_backend(tmp_path)
    cmd = (
        'file create --path /reports/decision.md --content "# Access Decision\n'
        '\n## Summary\nApprove the request.\n\n## Rationale\n- Minimal scope\n"'
    )
    res = backend.execute_cli(cmd)
    assert res.exit_code == 0, res.stderr

    body = backend.execute_cli("file read --path /reports/decision.md").stdout
    assert body.startswith("# Access Decision")
    assert "## Rationale" in body  # not truncated to the first line
    assert "\n" in body
    # The effect stream carries the full body for content checkers.
    created = res.state_changes["files_created"][0]
    assert created["content"] == body


def test_file_create_unescapes_literal_newlines(tmp_path) -> None:
    # Single-line content with literal ``\n`` escapes (the dominant Claude style)
    # must render as real newlines so it is not polluted with backslash-n tokens.
    backend = _fresh_backend(tmp_path)
    cmd = 'file create --path /r.md --content "# Title\\n\\n## Section\\nDone"'
    res = backend.execute_cli(cmd)
    assert res.exit_code == 0, res.stderr

    body = backend.execute_cli("file read --path /r.md").stdout
    assert body == "# Title\n\n## Section\nDone"
    assert "\\n" not in body


def test_file_create_preserves_literal_backslash(tmp_path) -> None:
    # A lone backslash before an unrecognized char (e.g. a regex ``\d``) is kept
    # verbatim; escape rendering only fires for the known sequences.
    backend = _fresh_backend(tmp_path)
    res = backend.execute_cli(r'file create --path /r.md --content "regex \d+ and path C:\dir"')
    assert res.exit_code == 0, res.stderr
    body = backend.execute_cli("file read --path /r.md").stdout
    assert r"\d+" in body
    assert r"C:\dir" in body


def test_file_create_preserves_inner_quotes_and_backticks(tmp_path) -> None:
    # Embedded quotes/apostrophes/backticks inside the content survive because the
    # value is peeled off before shlex tokenization.
    backend = _fresh_backend(tmp_path)
    cmd = (
        'file create --path /r.md --content "He said "yes" and `code` and it\'s ok"'
    )
    res = backend.execute_cli(cmd)
    assert res.exit_code == 0, res.stderr
    body = backend.execute_cli("file read --path /r.md").stdout
    assert '"yes"' in body
    assert "`code`" in body
    assert "it's ok" in body


def test_file_append_unescapes_literal_newlines(tmp_path) -> None:
    # The append path renders escapes the same way as create.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "report.md").write_text("Findings:", encoding="utf-8")
    backend = FileSystemBackend()
    backend.initialize(str(tmp_path), {"OPENCLAW_STATE_DIR": str(tmp_path)})

    res = backend.execute_cli('file append --path /report.md --content "\\n- root cause: v4.2\\n"')
    assert res.exit_code == 0, res.stderr
    body = backend.execute_cli("file read --path /report.md").stdout
    assert body == "Findings:\n- root cause: v4.2\n"
    assert "\\n" not in body

