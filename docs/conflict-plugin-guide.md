# Adding Your Own Conflict Plugin — Authoring Guide

This guide shows how to add a new conflict *mechanic* (a "conflict-type script
interface") to the `conflict_vlm_v1` generation line. A conflict plugin steers the
LLM generator to build a scenario around a specific coordination/temporal/
inconsistency mechanic, emit an expert trajectory that exercises it, and add the
checks that grade it.

A conflict plugin is **not** a runtime skill and **not** a simulator. It injects
per-type instructions into the generation pipeline's prompt hooks. If you want to
add a runtime *command family* (an MCP tool group / skill), read
[mcp-extension-guide.md](mcp-extension-guide.md) instead. For how the two surfaces
relate, see [How it maps to the rest of the system](#how-it-maps-to-the-rest-of-the-system).

The plugin surface is standardized the same way the skill/MCP surface is: every
plugin carries a declarative `ConflictPluginSpec` and self-registers via a
decorator. The declarative model lives in
`clasharness/scenario/plugins/spec.py` (the plugin analogue of
`clasharness/skills/spec.py`).

## The model in one sentence

```text
BaseConflictPlugin subclass (+ SPEC)  ->  @register_conflict_plugin  ->
imported in plugins/__init__.py  ->  _ALL_PLUGINS built from the registry  ->
get_plugins_for_taxonomy routes a seed's taxonomy tokens to it
```

A single `ConflictPluginSpec` declaration is the source of truth for the plugin's
routing metadata, conformance, and docs. It is metadata *around* the prompt hooks,
not a replacement for them: the five `get_*_instruction` hooks still carry the
actual prompt text the generator sees.

## Three steps

1. **Subclass `BaseConflictPlugin`** and attach a `SPEC`.
2. **Decorate the class with `@register_conflict_plugin`.**
3. **Make the module import** so the decorator runs — add one import line to
   `clasharness/scenario/plugins/__init__.py` (and to `__all__` if you want it
   publicly importable).

That's it. Importing the package runs every decorator, `_ALL_PLUGINS` is built from
the registry (in import order), and the token index is rebuilt so a seed whose
`conflict_taxonomy` / `difficulty_challenges` contains your token routes to your
plugin.

> **No packaging edit needed.** `clasharness` is discovered as a package, so a
> new plugin file is picked up automatically. The import in `__init__.py` is
> what actually registers it.

## Copy-paste template

This template models a coordination plugin that drives a skill family and asserts
its effects (the most instructive case — it exercises the cross-surface check). For
a pure-prompt plugin, omit `skill_families` / `emits_effects` (see
[Cross-surface effect check](#cross-surface-effect-check)).

```python
"""My custom conflict: <one line>."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class MyMechanicPlugin(BaseConflictPlugin):
  """Plugin modeling <the mechanic>."""

  # The single declarative source of truth (routing, conformance, docs).
  SPEC = ConflictPluginSpec(
      name="my_mechanic",                 # canonical taxonomy token + unique id
      summary="One-line description for docs / listings.",
      category="coordination",            # advisory; see spec.KNOWN_PLUGIN_CATEGORIES
      # taxonomy_tokens=("my_mechanic", "my_alias"),  # optional aliases; default (name,)
      skill_families=("lock",),           # command families the trajectory exercises
      emits_effects=("lock_acquired", "lock_released"),  # effects the evals assert
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  # `name` defaults to SPEC.name via BaseConflictPlugin; override only if you must.

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return "Plan phases covering: 1) ... and 2) ..."

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return "Model a subagent that ..."

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return "Create workspace files showing ..."

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- MY MECHANIC CHALLENGE: the trajectory MUST ..."

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- EVALS: add an effect check asserting ..."
```

Then add to `clasharness/scenario/plugins/__init__.py`:

```python
from clasharness.scenario.plugins.my_mechanic import MyMechanicPlugin
```

Now any seed with `"my_mechanic"` in its `conflict_taxonomy` /
`difficulty_challenges` routes to your plugin.

## The five prompt hooks

A plugin implements only the hooks it needs (each defaults to `""` and is skipped).
List the ones you implement in `SPEC.hooks` (advisory, but validated against the
closed `KNOWN_HOOKS` set). Each hook takes the `ConflictVLMSeed` and returns prompt
text merged into that generation layer.

| Hook | Generation layer | Typical content |
| --- | --- | --- |
| `get_layer0_instruction` | phase plan | the phases the scenario must cover |
| `get_layer1_instruction` | agent/persona | how a subagent must behave to create the conflict |
| `get_layer1b_instruction` | background workspace | files / chat evidence to embed |
| `get_trajectory_instruction` | expert trajectory | the exact steps the solver MUST take |
| `get_evals_instruction` | automated checks | the effect / file / command checks to add |

The core conflict plugins implement `layer0/layer1/layer1b`; the multimodal
"challenge" plugins implement `layer0/trajectory` (+ `evals`); the ten
coordination protocol plugins (`distributed_lock`, `barrier_synchronization`,
`controlled_deployment_gate`, `reversible_action_hold`, `multi_device_pairing`,
`scope_reconciliation`, `multi_party_synchronization`, `coordinated_containment`,
`non_overlapping_assignment`, `duplicate_hold_stale`) implement all five.

## `ConflictPluginSpec` fields

Defined in `clasharness/scenario/plugins/spec.py`:

| Field | Meaning |
| --- | --- |
| `name` | Canonical taxonomy token and unique plugin id (the primary match key). **Required.** |
| `summary` | One-line description for docs / listings. |
| `category` | Advisory grouping; see `KNOWN_PLUGIN_CATEGORIES` (`inconsistency`, `order`, `timeout`, `coordination`, `subagent_collaboration`, `main_supervised`, `challenge`). |
| `taxonomy_tokens` | All tokens that route to this plugin. Defaults to `(name,)`. Declare aliases here so a token rename doesn't silently match nothing. |
| `skill_families` | Command families (`agent_tools.v1.json`) the plugin's expert trajectory exercises, e.g. `("deploy",)`. Validated against the AgentToolSpec family set. |
| `emits_effects` | Effect keys the plugin's evals assert, e.g. `("deployment_cutover",)`. Cross-validated against the declared families' `state_effects`. |
| `hooks` | Which prompt hooks the plugin implements (subset of `KNOWN_HOOKS`). |
| `reused` | `True` if the mechanic predates the seed(s) that use it (metadata only). |

`SPEC.effective_tokens` returns `taxonomy_tokens` if set, else `(name,)` — this is
what routing uses.

## Cross-surface effect check

The interesting conformance rule (the plugin analogue of the skill-side
`CommandSpec.emits_effects` check) links the plugin surface to the **skill**
surface:

- Every family in `skill_families` must be a real `AgentToolSpec` family
  (`clasharness/specs/agent_tools.v1.json`, resolved via `normalize_family_name`).
- Every effect in `emits_effects` must be a `state_effect` that one of those
  families' commands can actually emit.
- Declaring `emits_effects` without any `skill_families` is an error.

This catches a plugin whose evals assert an effect no reachable command produces
(e.g. a `deploy`-family plugin claiming `lock_acquired`). If your plugin is
pure-prompt and does not drive a specific scaffolded effect check, leave both
`skill_families` and `emits_effects` empty — the cross-surface check then does not
apply. The twelve pure-prompt plugins (six core conflict + six multimodal
challenge) are in this category; the ten coordination protocol plugins declare
families/effects.

## Conformance & testing

`validate_plugins()` (in `plugins/__init__.py`) is the conformance checker; it is
asserted by `tests/test_conflict_plugin_spec.py` and returns a list of violations
(empty = OK). It checks, in order:

1. every plugin exposes a non-empty, unique `name`;
2. each attached `SPEC` is internally valid and `SPEC.name == plugin.name`
   (and, when `require_spec=True`, that a `SPEC` exists at all);
3. no taxonomy token is claimed by two different plugins;
4. the cross-surface `skill_families` / `emits_effects` check above.

Quick local check (pure stdlib; no build needed):

```bash
python3 -c "from clasharness.scenario.plugins import validate_plugins; \
print(validate_plugins(require_spec=True))"   # [] means conformant
```

Under the build:

```bash
pytest tests/test_conflict_plugin_spec.py
```

**Adding a plugin needs no new test** — `test_conflict_plugin_spec.py` covers every
registered plugin via `validate_plugins`. If you *do* add a plugin-specific test
file, put a `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
"..")))` before importing `clasharness` (see the top of
`tests/test_conflict_pipeline.py` / `tests/test_conflict_plugin_spec.py`).

## Rules & gotchas

- **Token uniqueness.** Two plugins must not claim the same taxonomy token;
  `validate_plugins` fails on a collision.
- **`name` == `SPEC.name`.** The property defaults to `SPEC.name`; if you override
  the property, keep it identical to the spec, or conformance fails.
- **Match by exact string.** A seed's `conflict_taxonomy` /
  `difficulty_challenges` token must equal one of the plugin's `effective_tokens`.
  A typo silently matches nothing (this bit us once: `db_audit` vs the plugin's
  real name `credential_lock`). Dump the routing tokens before authoring a seed:
  ```bash
  python3 -c "from clasharness.scenario.plugins import _ALL_PLUGINS, _effective_tokens; \
  print(sorted(t for p in _ALL_PLUGINS for t in _effective_tokens(p)))"
  ```
- **Registration is idempotent** — re-importing a module won't double-register the
  same class.
- **Import order = list order.** `_ALL_PLUGINS` preserves the import order in
  `__init__.py`; add your import where you want the plugin to appear.
- **A plugin only shapes generation.** It does not add a runtime command or an MCP
  tool. If your mechanic needs a new command family, build the skill first
  (see [mcp-extension-guide.md](mcp-extension-guide.md)) and then reference its
  family/effects from your plugin's `SPEC`.

## How it maps to the rest of the system

- Routing: `get_plugins_for_taxonomy` (taxonomy token → plugin) via the token index
  built from each plugin's `effective_tokens`.
- Conformance: `validate_plugins` — unique names, token collisions, spec validity,
  and the cross-surface effect check.
- Registration: `@register_conflict_plugin` (`plugins/registry.py`) +
  `build_registered_plugins()`; no hand-maintained list.
- Generation: the pipeline (`conflict_llm_pipeline.py`) calls the matched plugins'
  hooks to inject per-type instructions, then scaffolds checks over the generated
  trajectory (`_scaffold_round_checkers`).
- Skill surface: `skill_families` / `emits_effects` reference the runtime command
  families in `agent_tools.v1.json` that the plugin's generated trajectory
  exercises — that is the only link between the (generation-time) plugin surface
  and the (runtime) skill/MCP surface.

For the broader roadmap and the archetype → plugin mapping, see
[conflict-expansion-plan.md](conflict-expansion-plan.md).
