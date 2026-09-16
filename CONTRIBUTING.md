# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

---

## Project-specific development guidelines

CLASHarness is a benchmark-first repository centered on the multi-agent Conflict-VLM line (`conflict_vlm_v1`). Keep changes narrow, reproducible, and tied to real runtime behavior.

### Local setup

```bash
pip install -e .
pip install -e ".[dev]"
```

Optional Google-backed provider dependencies:

```bash
pip install -e ".[google]"
```

### Core commands

CLI help:

```bash
python examples/train_and_eval.py --help
python scripts/generate_conflict_vlm_scenarios.py --help
```

Targeted regression checks:

```bash
pytest -q tests/test_train_and_eval_cli.py
pytest -q tests/test_conflict_pipeline.py
pytest -q tests/test_conflict_vlm_reconcile.py tests/test_stateful_skill.py
```

Broader suite:

```bash
python -m pytest tests -v
```

### Development expectations

- Do not commit local secrets, OAuth tokens, or client-secret JSON files.
- Do not commit local trajectory dumps, temporary reports, or ad hoc benchmark outputs.
- Treat `clasharness/data/` as release content. If you regenerate tasks or datasets, describe exactly what changed and why.
- Keep public-facing docs aligned with the actual checked-in repo surface.
