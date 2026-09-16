# Adding Your Own MCP / Skill — Extension Guide

This guide shows how to add a new command family (an "MCP tool group" / skill)
to the arena **without editing backend wiring**. It uses the *additive
extension layer*: the built-in skills stay wired explicitly in
`MultiAppBackend`, and your skill is discovered, built, and registered
alongside them.

If you only want to understand the existing command surface, read
`docs/mcp-tooling-overview.md` first. For the declarative model every skill
uses, see `clasharness/skills/spec.py`.

## The model in one sentence

```text
Skill subclass (+ SPEC)  ->  @register_skill  ->  imported module  ->
MultiAppBackend builds it from a SkillContext  ->  SkillRegistry routes commands to it
```

A single `SkillSpec` declaration is the source of truth for the skill: it drives
command-routing registration, conformance checks, docs, and the dual-path tests.
(The live MCP server surface is curated separately in `backend/mcp_facade.py` —
see [How it maps to the rest of the system](#how-it-maps-to-the-rest-of-the-system).)

## Three steps

1. **Subclass `Skill`** (or `StatefulProtocolSkill` for `_emit`-style protocol
   effects) and attach a `SPEC`.
2. **Decorate the class with `@register_skill`.**
3. **Make the module import** so the decorator runs — either drop it into
   `clasharness/skills/contrib/` and import it from that package's
   `__init__.py`, or import your module yourself before constructing the
   environment.

That's it. `MultiAppBackend` calls `build_extension_skills(context)` at
construction and registers whatever it finds.

## Copy-paste template

```python
"""My custom skill: greet <name>."""

from __future__ import annotations

from clasharness.backend.base import CommandResult
from clasharness.skills import (
    CommandSpec,
    RealKind,
    Skill,
    SkillSpec,
    register_skill,
)


@register_skill
class GreetSkill(Skill):
    # The single declarative source of truth (routing, MCP facade, docs, tests).
    SPEC = SkillSpec(
        name="greet",
        category="compute",                 # advisory grouping; see spec.KNOWN_CATEGORIES
        summary="Greet a person by name.",
        prefixes=("greet",),                # first token(s) that route to this skill
        real_kind=RealKind.PURE_COMPUTE,    # how the real (non-mock) path is realized
        supports_mock=True,
        supports_real=True,
        commands=(
            CommandSpec(
                name="greet hello",         # must start with a declared prefix
                summary="Print a greeting.",
                args={"name": {"type": "string"}},
                required=("name",),
                emits_effects=(),           # effect keys this command may emit
            ),
        ),
        # Representative commands run (in order) by the dual-path smoke test.
        # Each must return exit_code == 0 from a fresh skill in mock AND real.
        smoke=("greet hello world",),
    )

    # Plain-data attributes that make up episode-mutable state (for `undo`).
    _snapshot_attrs = ("_greeted",)

    def __init__(self) -> None:
        super().__init__(prefixes=("greet",))
        self._greeted: list[str] = []
        self._real = False

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        # Per-episode setup. `state_dir` is the sandbox dir; env carries the
        # canonical real-mode signal (see below).
        from clasharness.backend.base import REAL_ENV_FLAG
        self._real = env_vars.get(REAL_ENV_FLAG) == "1"
        self._greeted = []

    def execute(self, command: str) -> CommandResult:
        parts = command.split()
        # parts[0] == "greet", parts[1] == subcommand
        if len(parts) >= 3 and parts[1] == "hello":
            name = parts[2]
            self._greeted.append(name)
            return CommandResult(stdout=f"Hello, {name}!", stderr="", exit_code=0)
        return CommandResult(stdout="", stderr="usage: greet hello <name>", exit_code=1)

    def get_state(self) -> dict:
        # Contributions surfaced to evaluators as environment state.
        return {"greeted": list(self._greeted)}

    def cleanup(self) -> None:
        pass
```

Drop this file at `clasharness/skills/contrib/greet_skill.py`, then add to
`clasharness/skills/contrib/__init__.py`:

```python
from clasharness.skills.contrib import greet_skill  # noqa: F401
```

Now every `MultiAppBackend` (and therefore `CLASHarnessEnv`) understands
`greet hello <name>`.

## `SkillContext`: getting shared dependencies

Some skills need collaborators that already exist in the backend (e.g. the
file backend). Those are handed to you at construction via a `SkillContext`.
Override `from_context`, or pass a `factory` to the decorator:

```python
@register_skill(factory=lambda ctx: IndexSkill(file_backend=ctx.file_backend))
class IndexSkill(Skill):
    def __init__(self, file_backend=None):
        super().__init__(prefixes=("index",))
        self.file_backend = file_backend
    ...
```

Equivalent, using `from_context`:

```python
@register_skill
class IndexSkill(Skill):
    @classmethod
    def from_context(cls, context):
        return cls(file_backend=context.file_backend)
```

`SkillContext` fields (all construction-time; per-episode `state_dir`/`env_vars`
still arrive via `initialize`):

| Field | What it is |
| --- | --- |
| `real_apps` | Whether the backend is in real-apps mode |
| `real_openclaw` | Whether the real curl/HTTP path is enabled |
| `strict_online_data` | Strict online-data mode flag |
| `file_backend` | `FileSystemBackend` (virtual workspace) |
| `message_skill` | `MessageSkill` (comms substrate) |
| `asset_backend` | `AssetBackend` |
| `http_backend` | `HttpBackend` |
| `agent_skill` | `AgentSkill` |
| `channel_skill` | `ChannelSkill` |
| `extras` | free-form dict for forward-compat / custom wiring |

## Mock vs real (`real_kind`)

`real_kind` tells the dual-path tests and docs how your skill's *real*
(non-mock) path behaves. Pick the closest:

| `RealKind` | Meaning | Examples |
| --- | --- | --- |
| `PURE_COMPUTE` | Deterministic local compute; real == mock | jq, grep, file |
| `LOCAL_PERSISTENT` | Real persists to local fs/sqlite; mock in-memory | lock, deploy, message |
| `EXTERNAL_NETWORK` | Real calls a network API; hermetic tests use a fake | calendar, email, curl |
| `SUBPROCESS` | Real shells out to a local binary | git, shell |
| `DETERMINISTIC` | Real == mock by design (no genuine impl in scope) | agent |

Read the canonical real-mode signal in `initialize` via
`REAL_ENV_FLAG` (`CLASHARNESS_REAL`) from `clasharness.backend.base`.

## Testing your skill

Use `extension_registry_sandbox()` so your test registration doesn't leak into
other tests sharing the process:

```python
from clasharness.backend.multi_app_backend import MultiAppBackend
from clasharness.skills import extension_registry_sandbox, register_skill

def test_greet_routes(tmp_path):
    with extension_registry_sandbox():
        register_skill(GreetSkill)
        be = MultiAppBackend()
        be.initialize(str(tmp_path), {})
        assert be._extension_errors == []
        res = be.execute_cli("greet hello world")
        assert res.exit_code == 0 and "Hello, world!" in res.stdout
        # SPEC conformance (routing/prefix/unique MCP tool names):
        assert be._skill_registry.validate_specs() == []
```

See `tests/test_skill_extensions.py` for the full set of examples (factory,
`from_context`, build-error isolation, prefix-collision handling).

## Rules & gotchas

- **Prefix uniqueness.** Your `prefixes` must not collide with an existing
  skill's. A collision is *collected* into `backend._extension_errors`
  (not raised) so a bad extension can't take down the core arena — but your
  skill won't be registered. Check `_extension_errors` if a skill is missing.
- **Every command name must start with a declared prefix** (enforced by
  `SkillSpec.validate()`).
- **Attach a `SPEC`.** Skills without one are flagged by
  `SkillRegistry.validate_specs()`. The spec drives conformance, routing
  metadata, and docs. (It does *not* generate the live MCP server surface —
  that comes from `backend/mcp_facade.py`; see "How it maps" below.)
- **`smoke` commands must pass from a fresh skill** in both mock and real
  modes (they back the dual-path test).
- **Registration is idempotent** — re-importing your module won't
  double-register the same class.
- **Build/registration failures are non-fatal** and surfaced via
  `backend._extension_errors`.

## How it maps to the rest of the system

- Routing: `SkillRegistry` (prefix → skill).
- Conformance: `SkillRegistry.validate_specs()` — this is the only consumer of
  `SkillSpec.to_mcp_tools()`, which it uses to enforce unique MCP tool names.
- MCP server surface: the live `tools/list` / `tools/call` comes from the curated
  `backend/mcp_facade.py` (served by `backend/mcp_server.py`), which is maintained
  **separately** from per-skill `SkillSpec`s. Adding a command to your `SPEC` does
  not add it to the MCP server; edit `mcp_facade.py` (regenerated via
  `scratch/gen_mcp_facade.py`) for that.
- Public API: everything you need is exported from `clasharness.skills`
  (`Skill`, `SkillSpec`, `CommandSpec`, `RealKind`, `register_skill`,
  `SkillContext`, `build_extension_skills`, `extension_registry_sandbox`, …).
