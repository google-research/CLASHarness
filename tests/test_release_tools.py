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

"""Guards for the release tooling under ``tools/``.

These three scripts decide what leaves the building, and until now nothing
tested them. Five of their properties are safety properties rather than
conveniences, and all five fail silently when they break:

  * ``scrub_corpus`` copies a *allowlist* of component directories taken from
    the ``download_data`` manifest. The corpus root it is pointed at also holds
    superseded eval trees, run provenance, staging scratch and a ``logs/``
    sibling beside every published ``tasks/``, and those carry far more operator
    detail than the published trees. A regression that widened the walk back to
    "everything under --src" would publish them redacted instead of not
    publishing them, and the only visible symptom would be a larger upload.

  * The exclusion list is matched as a substring. An equality test looks correct
    and lets ``_staging_minimax_m3`` straight through while appearing to exclude
    ``minimax_m3``.

  * The ``.error`` trim is what removes the internal diagnostic bodies, and
    almost nothing in them is matchable corpus-wide -- "groot" and "go/" are
    things scenario prose may legitimately say. If the trim stops firing, the
    corpus-wide detector will not catch what it leaves behind.

  * ``_maybe`` is a literal prefilter in front of every regex, and it is load
    bearing: without it a full scrub is hours rather than minutes. A pattern
    added without a corresponding literal turns the prefilter into a false
    negative, and a false negative here means the leak ships.

  * 86% of the corpus is base64 image payload, and a long random run reproduces
    a short identifier by chance -- this corpus really contains ``/BoRG/``
    inside a PNG, which ``\bborg\b`` matches under ``re.I``. Rewriting it
    corrupts the image, and nothing downstream would ever notice. Payload
    interiors are therefore skipped, and the margin that keeps the *edges* live
    is what stops the skip from hiding a real identifier that abuts one.

The rest of the file covers the ordinary contracts: redaction touches the
leaking string and nothing else, the manifests cannot drift apart, and a
corrupted download fails verification instead of being scored.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str):
  spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


download_data = _load("download_data")
scrub_corpus = _load("scrub_corpus")
checksum_corpus = _load("checksum_corpus")

EVAL_V8 = download_data.COMPONENTS["eval-v8"][0]
TASKS_V8 = download_data.COMPONENTS["tasks-v8"][0]


def _write(root: Path, rel: str, text: str) -> Path:
  path = root / rel
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")
  return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
  """A miniature corpus root: two published components and the traps."""
  root = tmp_path / "corpus"
  # Published.
  _write(root, f"{EVAL_V8}/aligned_setting_high/Claude/opus_5/r01/s.jsonl",
         json.dumps({"task_id": "t1", "success": True, "score": 1.0,
                     "error": None}) + "\n")
  _write(root, f"{TASKS_V8}/t1/specs.json", json.dumps({"task_id": "t1"}))
  # Not published: sibling of tasks/, superseded trees, provenance, staging.
  # The identifier *shapes* here have to be real or the detector will not fire;
  # the names inside them are placeholders, since this file is published.
  _write(root, "multiagent_conflict_data_v8_Gemini_3_6_flash/logs/run.log",
         "started on /bns/rt/borg/rt/job by testuser\n")
  _write(root, "eval_results/data_3_6_flash_v7/old.jsonl", '{"a": 1}\n')
  _write(root, "eval_results/_run_provenance/p.json",
         '{"workspace": "/google/src/cloud/testuser/x"}')
  # Both shapes matter. The inner directory is caught by an equality test too;
  # the batch script sitting at the top of `_staging_minimax_m3` is the one that
  # only a substring match keeps out, and in the real tree that top level is
  # where the scripts and provenance actually live.
  _write(root, "eval_results/_staging_minimax_m3/batch.sh",
         "#!/bin/sh\nrun --cell /bns/rt/borg/rt/job\n")
  _write(root, "eval_results/_staging_minimax_m3/MiniMaxAI/minimax_m3/x.jsonl",
         '{"a": 1}\n')
  return root


def _staged(dst: Path) -> set[str]:
  return {os.path.relpath(os.path.join(dp, fn), dst).replace(os.sep, "/")
          for dp, _dns, fns in os.walk(dst) for fn in fns}


def test_rel01_scrub_publishes_only_manifest_components(corpus, tmp_path):
  # REL-01: the allowlist holds -- siblings and superseded trees stay behind.
  dst = tmp_path / "staging"
  rc = scrub_corpus.cmd_apply(str(corpus), str(dst), brand=False,
                              prefixes=[EVAL_V8, TASKS_V8])
  assert rc == 0
  staged = _staged(dst)
  assert staged == {
      f"{EVAL_V8}/aligned_setting_high/Claude/opus_5/r01/s.jsonl",
      f"{TASKS_V8}/t1/specs.json",
  }
  # Named individually, because each is a different way of leaking.
  assert not (dst / "multiagent_conflict_data_v8_Gemini_3_6_flash"
              / "logs").exists()
  assert not (dst / "eval_results" / "data_3_6_flash_v7").exists()
  assert not (dst / "eval_results" / "_run_provenance").exists()


def test_rel02_staging_minimax_is_excluded_by_substring(corpus, tmp_path):
  # REL-02: `_staging_minimax_m3` != `minimax_m3`, so an equality test would
  # copy it. Checked through --whole-tree, where the allowlist is not helping.
  dst = tmp_path / "whole"
  scrub_corpus.cmd_apply(str(corpus), str(dst), brand=False, prefixes=None)
  staged = _staged(dst)
  assert not any("minimax_m3" in p for p in staged), staged
  # The rest of the tree did come across, so the assertion above is not vacuous.
  assert "eval_results/data_3_6_flash_v7/old.jsonl" in staged
  assert scrub_corpus._excluded("_staging_minimax_m3")
  assert scrub_corpus._excluded("minimax_m3")


def test_rel03_redaction_rewrites_the_error_and_nothing_else(tmp_path):
  # REL-03: the audited invariant -- a leaking `.error` on an already-failed
  # episode is rewritten, and every other byte on the line survives verbatim.
  src, dst = tmp_path / "s", tmp_path / "d"
  rec = {"task_id": "t9", "success": False, "score": 0.0,
         "final_response": "",
         "error": "connect to /bns/rt/borg/rt/uniserve-prod/0 failed",
         "trajectory": [{"step": 1, "stdout": "openclaw deploy status"}]}
  _write(src, f"{EVAL_V8}/r01/x.jsonl", json.dumps(rec) + "\n")
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])

  out = json.loads((dst / EVAL_V8 / "r01" / "x.jsonl").read_text())
  assert "/bns/" not in out["error"] and "uniserve" not in out["error"]
  assert "<cluster-address-redacted>" in out["error"]
  for key in ("task_id", "success", "score", "final_response", "trajectory"):
    assert out[key] == rec[key], f"{key} changed and must not have"
  # And --check now passes on the staged tree.
  assert scrub_corpus.cmd_check(str(dst), [EVAL_V8]) == 0


def test_rel04_clean_files_are_copied_byte_identical(tmp_path):
  # REL-04: a file with nothing to redact must come out unchanged, or every
  # digest in CHECKSUMS.txt would depend on the scrubber's JSON formatting.
  src, dst = tmp_path / "s", tmp_path / "d"
  raw = '{"task_id":  "t1",   "score": 1.0}\n{"task_id": "t2"}\n'
  _write(src, f"{EVAL_V8}/r01/clean.jsonl", raw)
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])
  assert (dst / EVAL_V8 / "r01" / "clean.jsonl").read_text() == raw


def test_rel05_identifier_outside_error_is_reported(tmp_path, capsys):
  # REL-05: the audit says these only ever appear in `.error`. If that stops
  # being true the tool must say so rather than widen the claim in silence.
  src, dst = tmp_path / "s", tmp_path / "d"
  _write(src, f"{EVAL_V8}/r01/y.jsonl",
         json.dumps({"task_id": "t", "final_response": "see /bns/rt/x"}) + "\n")
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])
  out = capsys.readouterr().out
  assert "WARNING" in out and ".final_response" in out


def test_rel06_missing_component_refuses_to_run(tmp_path, capsys):
  # REL-06: pointed somewhere that is not the corpus root, the tool must stop.
  # Staging nothing and reporting success is the dangerous outcome.
  with pytest.raises(SystemExit) as excinfo:
    scrub_corpus.main(["--src", str(tmp_path), "--check"])
  assert excinfo.value.code != 0
  assert "does not look like the corpus root" in str(excinfo.value.code)


def test_rel07_whole_tree_and_component_are_mutually_exclusive(tmp_path):
  # REL-07: the two scoping modes contradict each other; silently preferring
  # one would make a --component flag look honoured when it was ignored.
  with pytest.raises(SystemExit):
    scrub_corpus.main(["--src", str(tmp_path), "--check", "--whole-tree",
                       "--component", "eval-v8"])


def test_rel08_manifests_cannot_drift(tmp_path):
  # REL-08: scrub and checksum both read the download_data manifest rather than
  # restating it, which is the only reason the staged layout and the fetch
  # layout agree. A copy-pasted table would pass every other test here.
  assert scrub_corpus.PUBLISHED == {
      k: v[0] for k, v in download_data.COMPONENTS.items()}
  assert checksum_corpus.COMPONENTS is download_data.COMPONENTS
  assert set(download_data.DEFAULT) <= set(download_data.COMPONENTS)
  dests = [v[0] for v in download_data.COMPONENTS.values()]
  assert len(set(dests)) == len(dests), "two components share a destination"


def test_rel09_checksum_round_trip_and_corruption(corpus, tmp_path):
  # REL-09: a manifest written from a tree verifies against it, and a single
  # changed byte fails. This is the whole point of shipping CHECKSUMS.txt.
  manifest = tmp_path / "CHECKSUMS.txt"
  argv = ["--root", str(corpus), "--component", "eval-v8"]
  assert checksum_corpus.main(argv + ["--write", str(manifest)]) == 0
  assert checksum_corpus.main(argv + ["--verify", str(manifest)]) == 0

  victim = corpus / EVAL_V8 / "aligned_setting_high/Claude/opus_5/r01/s.jsonl"
  victim.write_text(victim.read_text() + " ", encoding="utf-8")
  assert checksum_corpus.main(argv + ["--verify", str(manifest)]) == 1


def test_rel10_checksum_is_sensitive_to_layout(corpus, tmp_path):
  # REL-10: the digest folds in the relative path, so a file moved within a
  # component changes it. Hashing contents alone would call a scrambled tree
  # identical to the published one.
  before, _n, _b = checksum_corpus.component_digest(str(corpus), EVAL_V8)
  src = corpus / EVAL_V8 / "aligned_setting_high/Claude/opus_5/r01/s.jsonl"
  src.rename(src.with_name("renamed.jsonl"))
  after, _n, _b = checksum_corpus.component_digest(str(corpus), EVAL_V8)
  assert before != after


# One example per identifier shape the tool knows about. Every row must be
# caught by the prefilter, or the regex behind it never runs.
_DOM = "google" + ".com"
_G3 = "google" + "3"
_SAMPLES = {
    "cluster-address": "connect /bns/rt/borg/rt/j failed",
    "workspace-path": "open '/google/src/cloud/testuser/x/defaults'",
    "batch-home": "cd /home/testuser_google_com/deploy",
    "local-home": "wrote /usr/local/google/home/testuser/out",
    "operator-compound": "user testuser_google_com",
    "operator-email": f"owner testuser@{_DOM}",
    "monorepo": f"built from {_G3} at head",
    "env-name": "activate openclaw_env now",
    "prod-host": f"peer 025rtbi1.prod.{_DOM} reset",
    "service-target": "to blade:cloud.ai.example.svc-prod-us",
    "internal-proto": "type.googleapis.com/example.orch.ErrorLoc",
    "vertex-model-id": "model /vertex/example-v4p1-s-rev24-lc-text-us",
    "debug-link": "see go/debugstr for detail",
    "job-name": "handled by stairways",
    "quota-id": "quota input-tpm:289334605194 exceeded",
}

# The two error bodies that actually leak, reduced to their shape. The real
# ones are 42 kB and 6 kB; what matters is the boundary, which is the first
# `{` -- everything before it is the human summary, everything after is the
# serialised payload.
_LEAKY_ERROR = (
    "504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline "
    "expired', 'details': [{'@type': 'type.googleapis.com/example.orch."
    "ErrorLoc', 'model': '/vertex/example-v4p1-s-rev24-lc-text-us', "
    "'target': 'blade:cloud.ai.example.svc-prod-us', 'host': "
    f"'025rtbi1.prod.{_DOM}', 'src': 'cloud/ai/platform/orcas/groot/"
    "service/handler.cc', 'help': 'go/debugstr'}]}}")
_CLEAN_ERROR = (
    'OpenAI-compatible request failed: HTTP 400: {"error":{"message":"This '
    'model\'s maximum context length is 65536 tokens.","code":400}}')


def test_rel11_prefilter_has_no_false_negative():
  # REL-11: the prefilter decides whether a regex runs at all, so a shape it
  # misses is a leak that ships. Each row is checked twice: the prefilter must
  # admit it, and some real pattern must then match it -- otherwise the row is
  # vacuous and proves nothing.
  for label, sample in _SAMPLES.items():
    assert scrub_corpus._maybe(sample), f"{label}: prefilter rejected it"
    matched = (scrub_corpus.DETECT.search(sample)
               or scrub_corpus._ERROR_PAYLOAD.search(sample)
               or scrub_corpus._QUOTA_ID.search(sample))
    assert matched, f"{label}: sample matches no pattern, so the row is vacuous"
  # A pattern or literal added without a row above would slip past the loop.
  assert len(scrub_corpus.REDACTIONS) == 14
  assert len(scrub_corpus._LITERALS) == 20
  assert len(_SAMPLES) == 15
  # And ordinary scenario prose must not be admitted, or the prefilter saves
  # nothing and every regex runs on all 49 GB.
  assert not scrub_corpus._maybe(
      "The deployment platform reported a failed rollout to the EU region.")


# A payload that looks like the real ones: base64 alphabet, no separator, long
# enough to be a run. The planted `/BoRG/` is the shape actually found in this
# corpus -- `/` is not a word character, so `\bborg\b` matches it under re.I.
_FILLER = "Qk1GRGh0V2FZS2VSeExraFpuUVhLVEpheFoxWjVSekVsNXdWU1o0RXpiT3JHZzNXRVlT"
_PAYLOAD = (_FILLER * 12)[:800] + "/BoRG/" + (_FILLER * 12)[:800]


def test_rel16_payload_interiors_are_left_alone(tmp_path):
  # REL-16: the filler must be innocent on its own, or the test proves nothing
  # about the planted match.
  assert not scrub_corpus._maybe((_FILLER * 12)[:800])
  assert scrub_corpus.DETECT.search(_PAYLOAD), "planted match does not fire"

  # Each layer on its own, because either one still skipping is enough to make
  # the end-to-end check below pass while the other has silently regressed.
  assert scrub_corpus.redact(_PAYLOAD) == _PAYLOAD, "payload was rewritten"
  assert not scrub_corpus.detect(_PAYLOAD), "--check would flag every image"
  assert not scrub_corpus.screen(_PAYLOAD.encode()), "byte screen let it through"

  src, dst = tmp_path / "s", tmp_path / "d"
  rec = {"task_id": "t6", "assets": {"a.png": {"__bytes_b64__": _PAYLOAD}}}
  _write(src, f"{EVAL_V8}/r01/b.jsonl", json.dumps(rec) + "\n")
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])

  out = json.loads((dst / EVAL_V8 / "r01" / "b.jsonl").read_text())
  assert out["assets"]["a.png"]["__bytes_b64__"] == _PAYLOAD
  # And --check must not report it either, or every image file is a finding.
  assert scrub_corpus.cmd_check(str(dst), [EVAL_V8]) == 0


def test_rel17_identifiers_abutting_a_payload_are_still_found(tmp_path):
  # REL-17: an identifier with no separator between it and a payload merges
  # into the run, so only the live margin at each end keeps it visible. Both
  # orders, because the margin is applied at both ends independently.
  before = "/bns/rt/j" + (_FILLER * 12)[:800]
  after = (_FILLER * 12)[:800] + "/bns/rt/j"
  for label, value in (("leading", before), ("trailing", after)):
    assert scrub_corpus.screen(value.encode()), label
    assert scrub_corpus.detect(value), label
    assert "<cluster-address-redacted>" in scrub_corpus.redact(value), label
  # The margin is what makes that work; it must stay wide enough for the
  # longest literal to straddle the boundary.
  assert scrub_corpus._BLOB_MARGIN == max(
      len(lit) for lit in scrub_corpus._LITERALS) - 1


def test_rel18_byte_screen_matches_the_text_screen():
  # REL-18: the file loop screens raw bytes and only decodes what survives, so
  # a disagreement between the two screens is a leak that is never decoded.
  for label, sample in _SAMPLES.items():
    assert scrub_corpus.screen(sample.encode()), f"{label}: byte screen missed"
    assert scrub_corpus.screen_text(sample), f"{label}: text screen missed"
  # Non-ASCII is the case the byte path could plausibly get wrong: it screens
  # before decoding, so a multi-byte sequence next to an identifier must not
  # hide it.
  spiced = f"caf\u00e9 \u2014 built from {_G3} \u2705"
  assert scrub_corpus.screen(spiced.encode())
  assert scrub_corpus.screen_text(spiced)
  # Case, which is the other thing the byte path could drop: every rule is
  # re.I, so a screen that compares raw case is a false negative.
  for shouty in (f"Built From {_G3.upper()} At Head", "peer Uniserve-04 reset"):
    assert scrub_corpus.screen(shouty.encode()), shouty
    assert scrub_corpus.screen_text(shouty), shouty
  assert len(scrub_corpus._LITERALS_B) == len(scrub_corpus._LITERALS)


def test_rel12_visualize_html_is_not_published(tmp_path):
  # REL-12: 3.8 GB of duplicated render that nothing reads, and a second copy
  # of every diagnostic string. Dropped by name, not redacted.
  src, dst = tmp_path / "s", tmp_path / "d"
  _write(src, f"{EVAL_V8}/r01/visualize.html", "<html>anything</html>")
  _write(src, f"{EVAL_V8}/r01/s.jsonl", '{"task_id": "t1"}\n')
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])
  assert _staged(dst) == {f"{EVAL_V8}/r01/s.jsonl"}


def test_rel13_error_payload_is_trimmed_not_redacted(tmp_path):
  # REL-13: the payload is dropped whole. Redacting it in place would leave
  # tens of kilobytes of placeholders, and most of what is in there -- job
  # names, source paths, debug links -- has no corpus-wide rule to redact with.
  src, dst = tmp_path / "s", tmp_path / "d"
  rec = {"task_id": "t9", "success": False, "score": 0.0,
         "error_type": "provider_failure", "final_response": "",
         "error": _LEAKY_ERROR,
         "trajectory": [{"step": 1, "stdout": "openclaw deploy status"}]}
  _write(src, f"{EVAL_V8}/r01/x.jsonl", json.dumps(rec) + "\n")
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])

  out = json.loads((dst / EVAL_V8 / "r01" / "x.jsonl").read_text())
  assert out["error"] == "504 DEADLINE_EXCEEDED. <internal-diagnostic-trimmed>"
  for bad in ("orcas", "groot", "/vertex/", "blade:", "go/debugstr",
              "type.googleapis.com", f".prod.{_DOM}", "handler.cc"):
    assert bad not in out["error"], bad
  # error_type is what analysis reads; the trim must not touch it, nor
  # anything else on the line.
  for key in ("task_id", "success", "score", "error_type", "final_response",
              "trajectory"):
    assert out[key] == rec[key], f"{key} changed and must not have"
  assert scrub_corpus.cmd_check(str(dst), [EVAL_V8]) == 0


def test_rel14_error_without_payload_survives(tmp_path):
  # REL-14: 493 of the 524 failed episodes have a useful error and nothing
  # internal in it. Trimming those would throw away why the run failed.
  src, dst = tmp_path / "s", tmp_path / "d"
  rec = {"task_id": "t8", "success": False, "score": 0.0,
         "error_type": "provider_failure", "error": _CLEAN_ERROR}
  _write(src, f"{EVAL_V8}/r01/y.jsonl", json.dumps(rec) + "\n")
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])
  out = json.loads((dst / EVAL_V8 / "r01" / "y.jsonl").read_text())
  assert out["error"] == _CLEAN_ERROR
  assert "maximum context length is 65536" in out["error"]


def test_rel15_quota_id_is_redacted_in_place(tmp_path):
  # REL-15: the rate-limit message names the project that was billed. The
  # number goes; the rest stays, because "this episode died on a rate limit"
  # is the one useful thing the string says.
  src, dst = tmp_path / "s", tmp_path / "d"
  err = ('OpenAI-compatible request failed: HTTP 429: {"error":{"code":'
         '"rate_limit_exceeded","message":"quota input-tpm:289334605194:'
         'openai.gpt-5.5 (InputTokens) exceeded by 4285"}}')
  rec = {"task_id": "t7", "success": False, "score": 0.0, "error": err}
  _write(src, f"{EVAL_V8}/r01/z.jsonl", json.dumps(rec) + "\n")
  scrub_corpus.cmd_apply(str(src), str(dst), brand=False, prefixes=[EVAL_V8])
  out = json.loads((dst / EVAL_V8 / "r01" / "z.jsonl").read_text())
  assert "289334605194" not in out["error"]
  assert "input-tpm:<quota-id-redacted>" in out["error"]
  assert "rate_limit_exceeded" in out["error"]
  assert "exceeded by 4285" in out["error"]
  # --apply ends by re-checking its own output, so --check has to know about
  # the quota id too; otherwise the substitution has no net under it.
  assert scrub_corpus.cmd_check(str(dst), [EVAL_V8]) == 0
  assert scrub_corpus.cmd_check(str(src), [EVAL_V8]) == 1


def _fake_hub(monkeypatch, published: set[str], seen: dict):
  """Install a huggingface_hub whose repo holds only the `published` prefixes."""
  import types

  def snapshot_download(repo_id, repo_type, local_dir, allow_patterns):
    seen["patterns"] = list(allow_patterns)
    for pattern in allow_patterns:
      prefix = pattern[:-len("/**")]
      if prefix in published:
        _write(Path(local_dir), f"{prefix}/part-0.json", "{}")
    return local_dir

  hub = types.ModuleType("huggingface_hub")
  hub.snapshot_download = snapshot_download
  monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_rel19_unpublished_component_fails_the_download(
    tmp_path, monkeypatch, capsys):
  # REL-19: huggingface_hub does not fail when a pattern matches nothing, so
  # the script has to. An existing but empty directory counts as missing too.
  root = tmp_path / "data"
  (root / TASKS_V8).mkdir(parents=True)
  monkeypatch.setattr(download_data, "DATA_ROOT", str(root))
  seen: dict = {}
  _fake_hub(monkeypatch, {EVAL_V8}, seen)
  rc = download_data.main(["--repo", "u/d", "--component", "eval-v8",
                           "--component", "tasks-v8"])
  err = capsys.readouterr().err
  assert rc == 1
  assert seen["patterns"] == [f"{EVAL_V8}/**", f"{TASKS_V8}/**"]
  assert "tasks-v8" in err and "eval-v8" not in err


def test_rel20_complete_download_succeeds(tmp_path, monkeypatch, capsys):
  monkeypatch.setattr(download_data, "DATA_ROOT", str(tmp_path / "data"))
  _fake_hub(monkeypatch, {EVAL_V8, TASKS_V8}, {})
  rc = download_data.main(["--repo", "u/d", "--component", "eval-v8",
                           "--component", "tasks-v8"])
  out = capsys.readouterr()
  assert rc == 0
  assert "Done." in out.out and out.err == ""
  # The hint names the directory the evaluator takes (the parent of tasks/),
  # and only for the task component.
  root = tmp_path / "data"
  hint = f"--task-data-dir {root / os.path.dirname(TASKS_V8)}"
  assert out.out.count("--task-data-dir") == 1 and f"{hint}\n" in out.out
  assert "analysis/" not in out.out


@pytest.mark.parametrize("argv", [
    ["--all", "--component", "eval-v8"],
    ["--default", "--component", "eval-v8"],
    ["--all", "--default"],
])
def test_rel21_selection_modes_are_mutually_exclusive(argv, capsys):
  # REL-21: --all or --default quietly winning over --component would fetch
  # something other than what was named.
  with pytest.raises(SystemExit) as excinfo:
    download_data.main(["--dry-run", *argv])
  assert excinfo.value.code == 2
  assert "not allowed with" in capsys.readouterr().err


def test_rel22_totals_are_summed_from_the_manifest(monkeypatch, capsys):
  # REL-22: the totals are derived, so they cannot drift from the table the
  # way a restated "N GB" did.
  monkeypatch.setattr(download_data, "COMPONENTS", {
      "a": ("x/a", "1.5 GB", ""), "b": ("x/b", "600 MB", ""),
      "c": ("x/c", "58 KB", ""), "d": ("x/d", "58 KB", "")})
  assert download_data._total(["a", "b"]) == "2.1 GB"
  assert download_data._total(["b", "c"]) == "600.1 MB"
  assert download_data._total(["c", "d"]) == "116 KB"
  assert download_data.main(["--dry-run", "--component", "b",
                             "--component", "c"]) == 0
  assert re.search(r"total\s+600\.1 MB", capsys.readouterr().out)
