# Visual & Web-Render Expansion Plan

Improving the **realism and diversity** of generated VLM imagery, giving pages on the
same platform **real underlying data relatedness**, and laying the groundwork for
future **web operations** (an agent that actually navigates/clicks a simulated web
app) in the `conflict_vlm_v1` line of CLASHarness.

> Status: **Phase VD complete — WD1–WD9 all landed** (SSoT archetype library, token-driven
> `compose_vlm_html`, deterministic seeded archetype+token sampling, the fixed no-LLM
> `chart_template.html` styling path, reproducibility hardening — seeded telemetry noise +
> the parallel `STYLE_PRESETS` retired in favor of the single SSoT style path — the
> archetype library expanded to 18 design systems (16 modern + 2 legacy) with stronger
> in-archetype variation, a quality-floor gate that rejects blank/collapsed renders
> across every local render path, and the WD8 test consolidation (cross-module SSoT
> single-source guards + end-to-end conflict-path styling reproducibility); all covered by
> unit tests, verified on mock/read-only paths — real Vertex/Chrome generation still owed).
> Next: M0–M5 of the conflict roadmap are built (only M6, scale & baseline, remains); then Phase VA (platform data model). This plan is the visual/web-element
> counterpart to [`docs/conflict-expansion-plan.md`](./conflict-expansion-plan.md):
> that doc owns *task mechanics, difficulty, and conflict archetypes*; **this doc owns
> the *visual surface* — how assets are rendered, how a platform's pages relate, and
> how the environment could become navigable.** Build order agreed with product:
> **D → A → C** (diversity first, then the platform data model, then interactive web).

---

## 0. Decisions (agreed)

| Decision | Choice |
|---|---|
| **Deliverable first** | **Full design doc before committing to a direction.** This document. |
| **Build order** | **D → A → C.** Rendering diversity (D) first — low-risk, no runtime change; then the platform data-model layer (A) for cross-page consistency; then interactive web + web-operation MCP (C), only if/when we commit to it. The three directions are **non-conflicting**; the only hard dependency is **C requires A**. |
| **VD approach** | **Option (b): LLM composes one coherent per-platform design language** from an SSoT archetype library — **not** static template expansion, **not** free per-axis random combination. Coherence via whole-token-set generation + per-platform lock + archetype seeding + compatibility rules + quality floor. See §3. |
| **Web-operations timing** | **Undecided — evaluate engineering cost first.** §5 exists to size C so the go/no-go can be made with numbers. |
| **Visual delivery for web ops (when C happens)** | **Step-by-step screenshot delivery** (re-render the current page to a PNG after each action and feed it to the VLM). Accepted that this requires runtime changes (`Observation` gains an image channel; `agents.py` delivers per-step, not once at priming). |

### Headline scope

| Lever | D (diversity) | A (data model) | C (interactive web) |
|---|---|---|---|
| Touches generation pipeline | ✔ | ✔ | ✔ |
| Touches runtime (`environment`/`Observation`/`agents.py`) | ✗ | ✗ | **✔** |
| New backend / navigable server | ✗ | ✗ | **✔** |
| New agent tool family | ✗ | ✗ | **✔** (`web.*`) |
| Depends on a prior phase | — | — | **A** |
| Effort (single-dev, w/ tests) | **M+** (~6–8 d) | **L** (~6–10 d) | **XL** (~15–25+ d, incl. A as its data floor) |

---

## 1. Background: how visual/web rendering works today

Five facts explain why the plan is shaped the way it is. Anchors are given as
function/symbol references because line numbers drift.

1. **Two front-ends share one render engine.** `vlm_materializer.py` (base) and
   `conflict_vlm_materializer.py` (conflict; calls `generate_vlm_assets` in
   `conflict_vlm_materializer.py:243`) both drive `CodeChartGenerator` in
   `vlm_asset_generators.py`.

2. **Rasterization is headless Chrome via subprocess**, not puppeteer/playwright.
   `vlm_asset_generators.py` shells out to `google-chrome --headless --screenshot`
   (`:149/231/738`); with no Chrome it degrades to matplotlib/PIL. Window is
   `CLASHARNESS_WINDOW_SIZE` (default `1920,1080`), scale `CLASHARNESS_RENDER_SCALE`
   (default `2.0`). Local libs (`tailwind.js`/`chart.js`) are copied from
   `templates/libs/` into a temp dir for offline rendering.

3. **Two template paths — production is fully LLM-authored HTML.**
   - *No-LLM fallback:* fixed `templates/chart_template.html` (injects
     `window.CHART_SPEC`) / `templates/text_template.html`.
   - *Production (with LLM):* the **entire HTML is written by the `compose_vlm_html`
     LLM layer** (`llm_client.py:2704-2746`); the fixed template is not used.
   `chart_template.html` (~1169 lines) hard-codes **2 layouts** (`devops_dashboard`,
   `single`) × **10 visual styles** (`modern_clean`, `retro_90s`, `flat_minimalist`,
   `terminal_tui`, `vercel_minimalist`, `datadog_dashboard`, `apple_developer`,
   `neo_brutalism`, `glassmorphism_3d`, `classic_mac_80s`). **8 `ui_shell_type`s**
   (ide / cloud_console / monitoring_dashboard / chat_app / database_manager /
   email_client / calendar_portal / general_web_page) collapse to those 2 layouts.
   Style is chosen by an **MD5 hash** of `title + platform_id + style_seed`
   (`vlm_materializer.py:820-828`) indexed into a suggestion list, which the LLM may
   override — i.e. **deterministic hash-seeded suggestion, not random.** A separate
   6-item `STYLE_PRESETS` (`conflict_vlm_materializer.py`, `kind==chart` only) used
   to inject a partial theme/color pair; **WD5 retired it** in favor of the single
   SSoT sampler.

4. **Same-platform pages share the *shell*, not the *data* — this is the core gap.**
   `platform_id` is a `VLMAssetSpec` field (`vlm_spec.py:141`), authored by the LLM in
   the asset-plan layer (`conflict_layer0c_assets_plan`, `llm_client.py:2183-2204`;
   requires ≥2 distinct web-UI surfaces), then normalized by
   `_sanitize_platform_id` / `_build_platform_vocab` / `_normalize_and_prefix_assets`
   (`conflict_llm_pipeline.py:2424-2605`; soft floor `_MIN_PLATFORM_VOCAB=2`).
   - **Shared today:** one `layout_spec` + `noise_spec` per platform
     (`generate_vlm_layout_spec`, `vlm_materializer.py:791-890`); `compose_vlm_html` is
     told to reuse `common_header/sidebar/footer_html`, shared CSS, `platform_context`
     (3–5 cross-screen-consistent strings like login_user/project_name), and the same
     grid ratios (`llm_client.py:2712-2742`).
   - **NOT shared (the defect):** the **underlying data / records.** Each asset's
     `chart_rows` / `ocr_text` / `description` / main panel is generated **one at a
     time, with no sibling awareness**, by the `vlm_asset_detail` layer
     (`vlm_materializer.py:382-513`, `llm_client.py:2601`); the layout designer cannot
     see asset data or siblings (`vlm_materializer.py:835-844`). ⇒ a list page's
     "order #1043" and a detail page's "order #1043" have **no mechanism** guaranteeing
     the same status/amount/customer.
   - **Image assets** (`natural_image`/`sketch`) are **fully independent**, excluded
     from platform sharing (`vlm_materializer.py:795`).
   - The only cross-asset tie is anchor **string tokens**; there is **no shared
     structured entity store.** (`consistency_layer.py` only reconciles *task-scoring*
     contradictions — it does **not** touch visual/data consistency.)

5. **Runtime consumes static PNGs, once.** The runtime feeds only static PNG pixels to
   the VLM, attached **once** in the episode-priming first user message
   (`train_eval/agents.py:81-138`, `:1296-1305`; runtime manifest is the answer-stripped
   `to_runtime_dict`, `vlm_spec.py:307-315`). `Observation` (`core/observation.py:18-57`)
   is **text-only — no image field.** In-environment `AssetBackend`
   (`backend/asset_backend.py`) returns only text metadata; `asset read-text` is a
   deliberate stub (the answer lives in pixels); `asset open` returns a path and records
   `assets_accessed`/`assets_opened`. There is **no browser/web/click/navigate/dom tool
   family**; the closest is `http_get` (a curl client — `http_backend.py` is a client,
   not a page server). There is **no routes/pages/navigation app spec**: an "app" today =
   ScenarioWorkspace (files) + ScenarioSession (chat) + ConflictScenarioUpdate
   (time-triggered updates) + protocol-skill state machines.

### Extension seams already available (reused by the phases below)

- **New effect bucket** rides the nested-effects passthrough in
  `environment.py:1233-1244` (comment: "no shipped producer sets `changes['effects']`")
  — no env change to record a new effect type.
- **New skill** → `skills/contrib/` + `@register_skill` (`multi_app_backend.py:153-180`).
- **New tool** → `backend/mcp_facade.py` (single source of truth, ~104 tools) + frozen
  goldens.
- **New tool family** → `specs/agent_tools.v1.json` + `tools/agent_tool_spec.py`.
- **Per-step visual/DOM delivery needs a new mechanism** (`Observation` is text-only).

---

## 2. Problem statement

| Product concern | Today | Root cause |
|---|---|---|
| Realism / diversity | Production is LLM-authored HTML, but boxed by 10 visual styles × 2 layouts × a deterministic hash seed; image assets are isolated | The render contract confines the possibility space to fixed enums |
| Same-platform depth | Only the shell is shared (chrome / theme / brand / `platform_context`); **no shared data model** | Asset detail is generated per-asset with no sibling awareness and no shared entity store |
| Future web operations | Absent: static PNG only, text-only `Observation`, no web tool family, no navigable backend | The system is a *screenshot benchmark*, not an *interactive environment* |

The three phases below address these in the agreed **D → A → C** order.

---

## 3. Phase VD — rendering diversity (direction D)

**Goal:** widen the visual possibility space and raise per-platform visual coherence,
**without** touching the runtime or the data layer. Lowest risk; ships visible
improvement first.

### Design principle — option (b): LLM composes a coherent per-platform design language

The chosen approach is **(b): at platform-creation time the LLM composes one whole,
coherent design language** (a complete token set) for that platform, sampled from a
**single source-of-truth (SSoT) archetype library**. It is explicitly **not**:

- **not** static template expansion (adding more baked `chart_template.html` branches), and
- **not** free per-axis random combination (independently rolling palette × type ×
  density × radius × shadow …), which produces incoherent, "not one real product" pages.

Coherence is guaranteed by five mechanisms working together:

1. **Whole-token-set generation** — a platform gets one internally-consistent token bundle
   at once, never axis-by-axis.
2. **Per-platform lock** — the bundle is fixed for the platform; every asset on it renders
   from the same tokens.
3. **Archetype seeding** — sampling starts from a curated archetype (a known-good design
   system), then varies *within* it.
4. **Compatibility rules** — archetype-declared constraints forbid clashing combinations.
5. **Quality floor** — a render-then-check gate rejects broken/empty output.

**Controlled-randomness granularity (resolves Open decision #3):** the seeded RNG picks
**an archetype + tokens within that archetype's declared ranges** — it does *not* shatter
the design into independent per-axis rolls. Same `style_seed` still reproduces exactly.

### Current-state findings that shape this plan

The `explore` pass mapped every consumer of `visual_style` / `style_details`:

- **The style vocabulary is duplicated across ~5 drifting places with no SSoT and no
  validation:** the prose enum in `llm_client.py:2669` + the per-style prose guide at
  `:2729-2738`; the real Python list of 10 in `vlm_materializer.py:798-809`; the JS
  `switch` (7 named branches) in `chart_template.html:305-691,1108-1151`; a second list in
  `generate_style_gallery.py:152-325`. Already drifting: a stale "11 vs 10" comment and an
  orphan token `enterprise_legacy` (`chart_template.html:318,344,425`).
- **No validation enforces the enum.** `_validate_parsed_data` (`llm_client.py:2752-2762`)
  only checks key presence; an unknown style name silently falls back to `modern_clean`.
- **Production diversity flows only through the LLM** (`compose_vlm_html`, prose-guided).
  The `chart_template.html` `switch` is (a) dashboard-layout only and (b) fed under a
  **mismatched key** in production (`styling` vs `custom_overrides.styling`,
  `vlm_materializer.py:1522-1547`), so outside the dev gallery it renders defaults — this
  is the **fallback bug WD9 fixes.**
- **A per-platform `style_details` token system already exists and is coherent** — this is
  the foundation option (b) builds on. `background_css_declaration` + `custom_css_rules`
  are generated but **not consumed** by the template (only the LLM prose path uses them).
- **Reproducibility:** `style_seed` (CLI flag, `scripts/generate_conflict_vlm_scenarios.py:126-127`)
  drives the seeded archetype+token sampler (WD2). The old leak —
  `_generate_telemetry_noise` using **unseeded** `random` so a pinned seed still jittered — is
  **fixed by WD5** (local, string-seeded RNG threaded from `style_seed`).
- **Two parallel style layers (now one):** the crude `STYLE_PRESETS` (theme+color only) was
  **retired by WD5**; the SSoT sampler is the single style path. `chart_presets.py`
  `default_styling` remains, but it is an **orthogonal matplotlib-fallback palette hint**
  (`theme: cool/warm`), not a competing template-styling source.

### Work items (WD1–WD9)

| # | Work item | Anchor | Effort | Risk | Status |
|---|---|---|---|---|---|
| WD1 | SSoT design-archetype library | new `clasharness/scenario/design_archetypes.py` | M | low | ✅ **done** |
| WD3 | Token-driven `compose_vlm_html` | `llm_client.py:2716-2739` | M | **med** | ✅ **done** |
| WD2 | Seeded archetype + token sampling | `vlm_materializer.py:820-828` | S–M | low–med | ✅ **done** |
| WD9 | Fix fallback key-mismatch bug | `vlm_materializer.py:1522-1547`, `chart_template.html` | S–M | low | ✅ **done** |
| WD5 | Reproducibility hardening | `vlm_materializer.py:1556-1592`; `conflict_vlm_materializer.py:1432-1459` | S | low | ✅ **done** |
| WD4 | Expand library to ~15–20 + strong in-archetype variation | `design_archetypes.py` | M | low | ✅ **done** |
| WD6 | Quality-floor render check | `vlm_asset_generators.py` render path | S–M | low | ✅ **done** |
| WD8 | Tests (reproducibility, SSoT single-source, fixtures) | `test_conflict_pipeline.py:626-654`, `test_llm_client.py:141-167` | M | low | ✅ **done** |
| ~~WD7~~ | ~~Give image assets the platform's visual tone~~ — **DROPPED** | — | — | — | — |

> **WD7 dropped (category error).** Standalone `natural_image`/`sketch` assets are
> physical-world photos (an implanted real device photo, `vlm_materializer.py:1159`; a
> hand-drawn whiteboard/paper sketch, `:1142,:1157`); tinting them with the platform's UI
> palette makes them *less* realistic. Digital `diagram` and in-HTML module images already
> inherit platform styling. The salvageable idea — that physical evidence should be
> *scenario-consistent across images* — is a **data** concern and moves to **Phase VA**
> (see §4).

#### WD1 — SSoT design-archetype library (M, low) — ✅ DONE

New module `clasharness/scenario/design_archetypes.py`. Each archetype is a frozen
`DesignArchetype` record: `name`, `token_preset` (the default `style_details`),
`palette_hints`, `domain_hints` (realism conventions), `legacy` flag,
`aesthetic_descriptor` (compact summary) + `render_guidance` (the detailed per-style prose
lifted verbatim from `compose_vlm_html`), `variation_ranges`, and `custom_css_rules`. It
exports the token-axis vocabularies (`STYLE_DETAIL_AXES`), `COLOR_THEMES`, `PRIMARY_COLORS`,
`UI_SHELL_TYPES`, `VISUAL_STYLE_NAMES` (historical order preserved), and accessors
(`get_archetype` w/ fallback, `archetypes_for_shell`, …). Now the one source that
**replaced the three drifting lists**: `vlm_materializer.py` (four vocab lists),
`generate_style_gallery.py` (its `styles_config`), and the prose enum in `llm_client.py`.
**Verified:** `tests/test_design_archetypes.py` (UT-ARCH-01..09) — canonical order stable,
all names resolve, presets valid against axes, presets **byte-match** historical values,
copy-on-read, fallback, per-shell legacy exclusion. Gallery presets byte-match; 412
existing tests still green.

#### WD3 — token-driven `compose_vlm_html` (M, med — core render contract) — ✅ DONE

The `compose_vlm_html` contract now renders **from tokens + injected design guidance**
instead of the 10 hardcoded per-name prose bullets. `vlm_materializer.py` injects a
`design_guidance` object (`{visual_style, aesthetic_descriptor, render_guidance}`) sourced
from `design_archetypes.get_archetype(layout_spec.visual_style)` into `compose_payload`; the
contract treats `render_guidance` as authoritative and falls back to a clean-modern
description if absent. The `visual_style` enum in `generate_vlm_layout_spec` is now generated
from `VISUAL_STYLE_NAMES` (`_visual_style_enum`) so it can never drift. Generic per-token
rules kept intact.
**Verified:** contract smoke (dynamic enum present, per-style bullets gone, `design_guidance`
referenced); 447 tests green. *Still owed: real Vertex/Chrome generation smoke (user-run).*

#### WD2 — seeded archetype + token sampling (S–M, low–med) — ✅ DONE

Replaced the MD5-index-into-flat-lists selection at `vlm_materializer.py:_design_platform_task`
with `design_archetypes.sample_platform_design(f"{title}_{pid}_{style_seed}")`. The sampler
uses `random.Random(seed)` (string-seeded → stable across runs/interpreters), picks a UI
shell, then an archetype that **fits that shell** (`archetypes_for_shell`, canonical order →
deterministic), then a coherent theme (new `theme_hints`/`allowed_themes()` — e.g.
`terminal_tui` is dark-only), a palette-coherent accent, and tokens varied only **within**
the archetype's `variation_ranges` (signature axes stay fixed). It emits a new
`suggested_style_details` in the layout payload; the contract instructs the LLM to use it as
the coherent baseline, and the mock generator now echoes it. Closed a coverage hole:
`calendar_portal` had no fitting archetype (added it to `modern_clean`/`flat_minimalist`/
`apple_developer`; also gave `ide` a modern option) so the incoherent full-pool fallback is
never hit. This intentionally **changes** output for existing `style_seed`s (better, coherent
sampling); the exit gate is *forward* determinism, not matching old MD5 output — acceptable
because VD precedes the M6 big generation.
**Verified:** `tests/test_design_archetypes.py` (UT-ARCH-10..16) — determinism (incl.
cross-process, `PYTHONHASHSEED`-independent), legality + coherence over 300–500-seed sweeps,
signature axes never drift, legacy excluded unless opted-in, `shell_hint` honored, full-library
+ both-theme diversity, every shell coherently covered. 483 tests green; end-to-end mock
wiring smoke passes.

#### WD9 — fix fallback key-mismatch bug (S–M, low) — ✅ DONE

**Root cause confirmed:** `chart_template.html:171` reads styling from
`spec.custom_overrides.styling` (shape `{theme, primary_color, visual_style, style_details}`),
but `_chart_spec_for_asset` emitted styling at the *top level* `chart_spec["styling"]` — which
the template never reads and the matplotlib fallback ignores. So every no-LLM template chart
silently rendered as default `modern_clean`/`blue`/`light` regardless of the platform.
**Fix:** new `_template_styling_for_asset(asset, spec)` resolves styling from the SSoT library
(`sample_platform_design`, deterministic per `(title, platform_id)` → one look per platform)
and both `_chart_spec_for_asset` return paths now attach it under
`chart_spec["custom_overrides"]["styling"]`. Only *template-valid* pre-set values are honored
(so the preset's matplotlib `theme: cool/warm` hint can't leak into the light/dark theme); an
explicitly named valid `visual_style` anchors its own coherent token preset. The top-level
`styling` (matplotlib hints) is left intact.
**Verified:** `tests/test_vlm_presets_overrides.py` (UT-WD9-01..06) — styling lands under
`custom_overrides.styling` for both preset and non-preset paths, coherent tokens, no
matplotlib-theme leak, per-platform determinism, explicit-style anchoring, full-styling
passthrough. 489 tests green. *Still owed: real Chrome render smoke (user-run).*
**Deferred to WD5:** thread the real `style_seed` into this path (currently seeds with 0).

#### WD5 — reproducibility hardening (S, low) — ✅ DONE

Three reproducibility/consistency fixes, all keyed off `style_seed`:
- **Seeded telemetry noise.** `_generate_telemetry_noise` used the unseeded module-global
  `random`, so a pinned `style_seed` still jittered the dashboard sparklines/metrics. It now
  takes `style_seed` and draws from a *local* `random.Random(f"{title}_{asset_id}_{style_seed}_telemetry")`
  — a string seed (version-2 seeding) that is `PYTHONHASHSEED`-independent and does not mutate
  global RNG state. `style_seed` is threaded through `_chart_spec_for_asset` (new optional arg,
  default 0) from the `generate_vlm_assets` caller.
- **Retired the parallel `STYLE_PRESETS`.** The 6-item `{theme, primary_color}` table in
  `conflict_vlm_materializer._generate_fresh_spec` (chosen by `hash(slug+pid) % 6`) set only 2 of
  the 4 styling axes and could clash with the platform's archetype (it partially overrode the
  coherent WD9 base at render time). It is replaced by one coherent SSoT bundle per platform
  (`design_archetypes.sample_platform_design(f"{slug}_{pid}_{style_seed}")` → theme + primary_color
  + visual_style + style_details). There is now a **single style path** (the SSoT sampler) shared
  by the base design path, the no-LLM `chart_template.html` fallback, and the conflict path.
  `style_seed` is threaded into `_generate_fresh_spec`.
- **Closed the WD9 deferral.** `_template_styling_for_asset` no longer hard-codes `style_seed=0`;
  it takes the real `style_seed` and samples on `f"{title}_{pid}_{style_seed}"` — the exact key the
  LLM design path (`_design_platform_task`) uses, so the fallback and design paths agree for a
  pinned seed.
- **Deduped a second drifting shell list.** The hardcoded 8-shell `suggested_shells` /
  `["light","dark"]` used to seed the layout-spec *fallback* now come from
  `design_archetypes.UI_SHELL_TYPES` / `COLOR_THEMES`, so the shell/theme enums can't drift.

**Verified:** `tests/test_vlm_presets_overrides.py` (UT-WD5-01..05) — telemetry noise is
deterministic for a pinned seed, varies with seed/asset, and is **byte-identical across a fresh
interpreter** (`PYTHONHASHSEED`-independent, digest `8d560ecc…` under HASHSEED 1 and 99);
`_template_styling_for_asset` honors `style_seed` and matches the design-path key;
`_chart_spec_for_asset` threads the seed into both telemetry and styling.
`tests/test_conflict_pipeline.py::TestConflictResume::test_platform_style_is_coherent_and_seed_reproducible`
— the conflict path now emits a *coherent* full bundle (visual_style + style_details, impossible
under `STYLE_PRESETS`), per-platform-consistent and reproducible per `style_seed`. The pre-existing
`test_platform_style_consistency` still passes unchanged. Full suite green (1410 passed; the single
unrelated failure — `test_environment.py … check_system_status_1` — is broken at HEAD, a
task/checker fixture unrelated to VD). *Still owed: real Chrome render smoke (user-run).*

#### WD4 — expand the library (M content, low) — ✅ DONE

Grew the archetype library to **18 design systems** (was 10): **16 modern non-legacy +
the 2 unchanged legacy looks**, all with **strong in-archetype variation**. Cheap because
WD3/WD9 made rendering token-driven — no new template branches required.

**What was built** (`design_archetypes.py`):
- **8 new modern archetypes appended** after the historical 10 (order preserved so
  pre-existing MD5-seeded selections stay reproducible): `material_design`, `notion_docs`,
  `linear_dark`, `stripe_dashboard`, `github_dark`, `bloomberg_terminal`, `spotify_media`,
  `discord_chat`. Each is a full coherent bundle: `token_preset` (all 9 axes),
  `palette_hints` (default color ∈ palette, all ∈ `PRIMARY_COLORS`), `domain_hints`
  (realistic shell fit), `theme_hints`, `aesthetic_descriptor`, detailed `render_guidance`
  (SSoT-injected into `compose_vlm_html` via `design_guidance`), and 2–5 coherent
  `variation_ranges` axes.
- **Stronger in-archetype variation** on existing archetypes (`modern_clean`,
  `flat_minimalist`, `datadog_dashboard`, `apple_developer`, `neo_brutalism`,
  `glassmorphism_3d`) — extra coherent knobs (dividers, header style, texture, radius,
  shadow) so same-archetype platforms don't collapse onto one look.
- **Per-shell variety widened**: every UI shell now has ≥3 coherent non-legacy archetypes
  (e.g. `ide` 2→4, `monitoring_dashboard` 3→7, `general_web_page` 6→13).

**Why it was safe / free:** new archetypes flow automatically through the SSoT — the
gallery iterates `all_archetypes()`, `llm_client._visual_style_enum` is generated from
`VISUAL_STYLE_NAMES`, and `design_guidance` is read from `get_archetype(...).render_guidance`.
The `chart_template.html` no-LLM path applies **generic token treatment** (sections 1–7)
for any `visual_style`; the name-keyed signature flourishes (lines 466–691) are an
`if/else if` chain with **no final `else`**, so a novel name gracefully skips bespoke
flourishes and still gets full token styling.

**Verified:** `tests/test_design_archetypes.py` — 18 tests pass, incl. UT-ARCH-16
(the 500-seed sampler exercises **exactly** the 16-name non-legacy set + both themes),
new UT-ARCH-17 (library expanded, historical order stable, legacy set unchanged, every
new archetype reachable) and UT-ARCH-18 (every shell has ≥3 coherent archetypes). Gallery
generation smoke (`scripts/generate_style_gallery.py`) emits all 18 `style_*.html` from
the SSoT. Full suite: 1412 passed + 31 subtests (one unrelated HEAD-broken
`test_environment.py::check_system_status_1` checker fixture). *Still owed: real
Vertex/Chrome render smoke across the new archetypes (user-run).*

#### WD6 — quality-floor render check (S–M, low) — ✅ DONE

Render-then-check gate so loosened sampling / a broken token bundle cannot silently
ship a *collapsed* (zero/degenerate-dimension) or *blank* (single flat color / no ink)
page. Landed in `vlm_asset_generators.py`:

- **Pure measurement** `_render_quality_report(data)` → `RenderQualityReport`: decodes
  the PNG (undecodable bytes are a hard failure), enforces a `_MIN_RENDER_DIM = 16`px
  floor on each side, then estimates ink resolution-independently — alpha is composited
  over opaque white (so a transparent page reads as blank, not as one "transparent"
  color), the image is downscaled to ≤200×200, and a render is "blank" if it has ≤1
  distinct color or fewer than `_MIN_INK_FRACTION = 0.002` (0.2%) non-background pixels.
  Pages with more colors than the counting cap short-circuit as unambiguously rich.
- **Enforcement** `_assert_render_quality(data, *, context)` raises `RenderQualityError`
  (tagged with the render context) unless `CLASHARNESS_DISABLE_RENDER_QUALITY_CHECK=1`.
- **Wired into all five local render paths**: chart HTML/Chrome, text HTML/Chrome,
  chart Matplotlib, python-subprocess, and raw-LLM-HTML. A tripped gate on the chart
  HTML path falls back to Matplotlib (existing `render_chart_spec` try/except); a
  tripped gate on the raw-HTML compose path propagates up to `vlm_materializer` as a
  `ScenarioLLMError`, rejecting the broken bundle rather than materializing it. The
  two *external* image providers (Google / Vertex) are intentionally out of
  scope — the gate guards our own render path, not model-generated imagery.

**Tests:** `tests/test_render_quality.py` (UT-RQ-01…11) builds synthetic PNGs with PIL —
blank white, single flat color, degenerate dimensions, undecodable bytes, fully
transparent, rich content, minimalist-but-sufficient ink, determinism, context-tagged
raise, healthy pass-through, and the escape-hatch bypass. `test_flexible_chart_parser.py`
(real Matplotlib/Chrome renders) confirms healthy charts pass the gate.
**Verify:** WD6 + chart-parser suite `19 passed`; full suite `1423 passed + 31 subtests`
(only the pre-existing HEAD-broken non-VLM `check_system_status_1` fails).
Real Vertex/Chrome generation smoke still owed (user-run).

#### WD8 — tests (M, low) — ✅ DONE

Consolidation pass: much of WD8 landed incrementally alongside WD1–WD6, so this item
audited the coverage, confirmed the mock fixtures were already on the token contract,
and filled the remaining cross-module gaps.

- **Already covered (verified, not re-done):** deterministic reproducibility of the
  sampler (`test_design_archetypes.py` UT-ARCH-12/13/16 — same seed → byte-identical
  design, coherent/legal across 300 seeds, full-library diversity across 500) and of the
  materializer seeding (`test_vlm_presets_overrides.py` UT-WD5-01…05 — telemetry noise +
  template styling keyed on `style_seed`, cross-process stable, matches
  `sample_platform_design`). The `generate_vlm_layout_spec` mock fixture
  (`test_conflict_pipeline.py:626-654`) already emits the full 9-axis `style_details` +
  `visual_style` passthrough, and `test_llm_client.py:141-167` (JSON terminal-degeneration
  recovery) is current — no fixture updates were needed.
- **New — SSoT single-source guards** (`tests/test_wd8_ssot_consistency.py`, UT-WD8-01…05):
  the style vocabulary has exactly one source (`VISUAL_STYLE_NAMES` == `archetype_names()`
  == the archetype objects, no dupes); the retired parallel sources stay retired (no
  `STYLE_PRESETS` / `suggested_visual_styles` attribute on either materializer or
  `llm_client`, and all three import the SSoT module); the injected `design_guidance` is
  SSoT-sourced and populated (non-stub `aesthetic_descriptor` + `render_guidance` for every
  style); `llm_client`'s compose enum is generated from `design_archetypes.VISUAL_STYLE_NAMES`;
  the gallery iterates `design_archetypes.all_archetypes()`.
- **New — end-to-end conflict-path styling** (`tests/test_conflict_pipeline.py`, UT-WD8-06/07):
  a mock-generation run of `materialize_conflict_scenario` (no network/Chrome) confirms every
  chart asset carries a coherent, library-valid style bundle in `custom_overrides.styling`
  (signature axes match the archetype preset — no cross-archetype token bleed) equal to what
  the SSoT sampler yields for the platform key, and that a pinned `style_seed` reproduces the
  whole platform look through the full pipeline while the seed remains a real knob.

**Verify:** WD8 subset `8 passed`; full suite `1430 passed + 31 subtests` (only the
pre-existing HEAD-broken non-VLM `check_system_status_1` fails). Real Vertex/Chrome
generation smoke still owed (user-run).

### Suggested implementation order

**WD1 → WD3 → WD2 → WD9 → WD5 → WD4 → WD6 → WD8.** (Land the SSoT and the token-driven
contract first, then sampling, then fix the fallback, harden reproducibility, expand the
library, add the quality gate, and finish with tests.)
All work items **WD1–WD9 have landed** — Phase VD is complete. Next up per the agreed
program order: the conflict roadmap's M0–M5 are built (only M6, scale & baseline, remains),
then Phase VA (platform data model).

### Files

New `clasharness/scenario/design_archetypes.py`; `llm_client.py` (`compose_vlm_html`
contract `:2716-2739`, prose enum `:2669`, per-style prose `:2729-2738`, realism
conventions `:2656-2663`, validation `:2752-2762`); `vlm_materializer.py` (sampling
`:820-828`, archetype list `:798-809`, fallback key `:1522-1547`, telemetry noise
`:1556-1592`); `templates/chart_template.html` (fallback grammar + key read);
`generate_style_gallery.py:152-325`; `conflict_vlm_materializer.py` (`STYLE_PRESETS:1432-1459`);
tests `test_conflict_pipeline.py:626-654`, `test_llm_client.py:141-167`.

### Effort / risk / impact

- **Effort:** M+ (~6–8 days).
- **Risks:** WD3 changes the core render contract (mitigate: generation smoke, staged
  rollout); loosened sampling can hurt **reproducibility** and **scoring controllability**
  (mitigate: seeded archetype+token RNG, WD5 noise seeding, WD6 quality floor); free
  combination would look incoherent (mitigated by design — whole-token-set + archetype
  seeding + compatibility rules).
- **Six-dim impact:** realism ↑↑ / diversity ↑↑↑ / cross-page consistency ↑ (per-platform
  coherence) / future web ops (no direct help) / generation cost → / scoring
  controllability → (seeded RNG + SSoT + validation compensate).

### Exit gate

- ✅ Same `style_seed` reproduces identical archetype + tokens (seeded RNG unit test) and
  byte-identical telemetry noise (WD5 / UT-WD5-*, UT-ARCH-12).
- ✅ One vocabulary source only; no unknown-style silent fallback (SSoT test /
  UT-WD8-01…05, UT-ARCH-08).
- ✅ Fallback path renders the selected archetype (WD9 / UT-WD9-*); quality-floor check
  green (WD6 / UT-RQ-01…11).
- ⬜ Generate one existing + one Vertex seed: visibly broader **yet per-platform coherent**
  style spread with **0** render failures. *(owed — needs real Vertex/Chrome, user-run.)*
- ✅ Full regression suite green (`1430 passed + 31 subtests`; only the pre-existing
  HEAD-broken non-VLM `check_system_status_1` fails).

---

## 4. Phase VA — platform data-model layer (direction A)

**Goal:** give same-platform pages **real data relatedness** — the fix for concern #2 —
and, as a side effect, establish the **data floor that Phase VC (web ops) requires
(= C0).** Still a screenshot benchmark; changes are confined to generation time.

### Design

1. **New "platform data-model" generation layer**, run after asset-plan and before
   asset-detail. Produces one `platform_data_model` per platform: a structured JSON
   entity set (e.g. `orders[]`, `users[]`, `deployments[]`, `records[]`) whose *schema*
   is driven by scenario semantics (may be empty for pure-image platforms —
   **[Open decision #2]**). Schema authored by LLM vs. from a semi-structured entity-type
   library is **[Open decision #1]**.
2. **Asset detail projects from the model instead of inventing data.**
   `vlm_asset_detail` (`llm_client.py:2601`, `vlm_materializer.py:382-513`) changes from
   "freely generate rows/text" to "**select a view/slice of `platform_data_model`**".
   Each asset declares which entities/fields/view (list / detail / chart) it renders.
3. **Break the information isolation.** Inject the platform's model summary into both the
   layout designer (`vlm_materializer.py:835-844`) and asset-detail prompts.
4. **Consistency calibration at generation time.** Add a C2-style pass that asserts every
   same-platform asset's rendered entity values match the model (reuse the calibration
   pattern already in `conflict_vlm_materializer.py`). Anchors should be **derived from
   the model**, not a parallel source.
5. **Physical-evidence cross-image consistency (salvaged from dropped WD7).** Standalone
   `natural_image`/`sketch` assets stay physical-world photos (they must **not** inherit UI
   tone — see §3 WD7-dropped), but the *scenario facts they depict* (a device's serial /
   error code / handwritten order number) should agree with the platform data model and
   with each other. Feed the relevant model slice into the image-prompt layer
   (`vlm_materializer.py:1131-1194`) so physical evidence is **data-consistent**, not just
   independently invented. This is the data half of the old WD7 idea.
6. **Graceful fallback:** if the model layer fails/returns empty, fall back to today's
   per-asset generation (no regression for platforms that opt out).

### Files

`llm_client.py` (new layer + `vlm_asset_detail` contract), `vlm_materializer.py`
(scheduling order, model injection, isolation break `:835-844`), `conflict_llm_pipeline.py`
(hook at platform-normalization `:2424-2605`), `vlm_spec.py` (new `PlatformDataModel`
type; `VLMAssetSpec` gains a `data_view` reference), `conflict_vlm_materializer.py`
(consistency calibration).

### Effort / risk / impact

- **Effort:** L (~6–10 days).
- **Risks:** a new LLM layer raises generation cost + failure modes (mitigate: fallback);
  an over-rigid schema can **suppress diversity** — everything looks like a CRUD admin
  (mitigate: scenario-driven, optional schema); interaction with anchors/`consistency_layer`
  must be untangled (anchors derive from the model).
- **Six-dim impact:** realism ↑↑ / diversity → (↑ if schema is scenario-driven, ↓ if not)
  / cross-page consistency ↑↑↑ / future web ops: **this is the data floor (C0)** /
  generation cost ↑ / scoring controllability ↑ (values deterministically derivable).

### Exit gate

- Same-platform assets referencing an entity render **identical** field values
  (calibration test); a deliberately mismatched fixture is caught.
- Generate a multi-page platform seed; list/detail pages agree on shared records.
- Pure-image platforms unaffected; fallback path exercised. Full suite green.

---

## 5. Phase VC — interactive simulated web + web-operation MCP (direction C)

**Goal (only if/when committed):** turn the screenshot benchmark into a **navigable
simulated web app** — declarative `routes/pages/state`, a backend that serves navigable
pages, a `web.*` tool family, and **step-by-step screenshot delivery** to the VLM. **C is
a superset of A:** navigable pages must be driven by A's shared data model.

**Framework choice — self-built lightweight spec, not a heavyweight FE framework.**
Running a real React/Vue app introduces runtime/build/hosting deps that don't fit the
"Chrome-screenshot + LLM-authored HTML" pipeline and are hard for an LLM to author
reliably. Recommend a **declarative spec** (route → template → data-model slice + a small
state machine) reusing the existing Chrome rasterizer. (Swapping the
`google-chrome --screenshot` subprocess for playwright is an *implementation option* for
stable per-step interaction/capture in C6/C7 — weigh new dep vs. subprocess fragility.)

### Engineering breakdown (for the go/no-go)

| # | Work item | Notes | Size | Regression risk |
|---|---|---|---|---|
| C0 | Data-model floor (= Phase VA) | navigable pages need a stable data source | L | med (gen-time) |
| C1 | Simulated web spec types | declarative routes/pages/interactive-elements/initial state | M | low (new types) |
| C2 | Navigable backend | route+state → HTML → Chrome screenshot; state-transition engine (extend `multi_app_backend.py`) | L | med |
| C3 | `web.*` tool family | `navigate/click/type/back/read_dom` × (json spec + facade + tool spec) | M | low–med |
| C4 | Effect tracking + scoring | record `page_visited`/`clicked`/`form_submitted` via nested-effects passthrough (`environment.py:1233-1244`); new checker | M | med |
| C5 | **`Observation` gains an image channel** | touches the core runtime data structure (`core/observation.py:18-57`) | M | **high** (all adapters) |
| C6 | **`agents.py` per-step screenshot delivery** | re-render current page each step, deliver as image part (today: only priming `:1296-1305`); needs screenshot cache/dedupe | L | **high** |
| C7 | Per-step render performance | Chrome reuse / page cache / concurrency so each step isn't a cold start | M | med |

- **Total:** XL (~15–25+ days, **including A as C0**). Highest-risk items are **C5/C6**
  (core runtime + every agent adapter); **C0 = Phase VA is the non-skippable prerequisite.**

### Files

New web-spec module (or `vlm_spec.py`), new/extended backend (`backend/multi_app_backend.py`,
`backend/asset_backend.py`), `backend/mcp_facade.py` + goldens, `specs/agent_tools.v1.json`,
`tools/agent_tool_spec.py`, `core/environment.py` (effect bucket + per-step render hook),
`core/observation.py` (image channel), `train_eval/agents.py` (per-step delivery), a new
effect checker under `evaluation/checkers/`. **Depends on Phase VA's data model as the page
data source.**

### Effort / risk / impact

- **Effort:** XL (see table).
- **Risks:** `Observation` image channel + per-step delivery touch runtime core and all
  agent adapters (wide regression surface); per-step Chrome rendering raises cost/latency
  (mitigate: cache/reuse unchanged pages); scoring shifts from "final answer" to
  "interaction trajectory" (needs new checkers); an interactive state machine is a new
  correctness/reproducibility burden.
- **Six-dim impact:** realism ↑↑↑ / diversity ↑ / cross-page consistency ↑↑↑ / future web
  ops: **this is the goal** / generation+runtime cost ↑↑↑ / scoring controllability ↓
  (interaction adds nondeterminism; constrain deliberately).

### Exit gate (when built)

- A seed exposes ≥2 navigable routes driven by one data model; `web.navigate`/`web.click`
  reach the expected page and emit `page_visited`/`clicked` effects.
- Per-step screenshot reflects post-action state; deterministic replay of an
  action sequence yields identical screenshots.
- Runtime regression suite green across all agent adapters.

---

## 6. Ordering rationale (why D → A → C works)

- **Non-conflicting:** D is purely *visual* (style/layout/seed); A is *data*; C is
  *interaction*. D and A/C are orthogonal; edits overlap only as **additive** touches to
  the same files (`llm_client.py`, `vlm_materializer.py`) in **different functions** — no
  rework of D when A lands.
- **Only hard dependency:** **C requires A** (C0). D → A → C respects it.
- **Accepted interim state:** after D-only, pages are *more diverse but still
  data-inconsistent* across the same platform; A closes that later. This is expected, not
  a regression.
- **Alternative considered:** A-before-D guarantees self-consistent data at every moment,
  but D-first delivers a low-risk, runtime-free win and validates "loosen constraints
  without losing quality." Product chose D-first; both are valid.

---

## 7. Effort & risk summary

| Phase | Effort | Runtime change | Key risk | Delivers even if later phases are dropped |
|---|---|---|---|---|
| VD (diversity) | M+ (~6–8 d) | none | core render-contract change (WD3); reproducibility / scoring control | ✔ concern #1 (realism/diversity) + per-platform visual coherence |
| VA (data model) | L (~6–10 d) | none | schema rigidity vs. diversity; gen cost | ✔ concern #2 (cross-page relatedness) + web-ops data floor |
| VC (interactive web) | XL (~15–25+ d, incl. A) | **yes** (`Observation`, `agents.py`) | wide runtime regression; per-step cost | ✔ concern #3 (web operations) |

If web operations are never built, **VD + VA alone resolve concerns #1 and #2** with no
wasted work.

---

## 8. Open design decisions (need product input)

1. **Data-model schema authorship (VA):** fully LLM-generated per scenario semantics
   (more general, less controllable) *vs.* a semi-structured **entity-type library** the
   LLM selects from (more controllable, may cap diversity)? **[OPEN]**
2. **Image assets (VD/VA):** ~~do `natural_image`/`sketch` assets join the platform visual
   tone?~~ **Resolved for VD — no** (physical photos keep their real look; WD7 dropped).
   The live question is narrower and now lives in VA: should physical-evidence images be
   made **data-consistent** with the platform model (§4 item 5), or stay independent?
3. **Controlled randomness (VD):** ~~accept seeded RNG for style?~~ **RESOLVED — yes, and
   shipped (WD2).** Seeded RNG picks **an archetype + tokens within its declared ranges**
   (not per-axis free rolls); the same `style_seed` reproduces **exactly and across process
   restarts** (`PYTHONHASHSEED`-independent). Note the guarantee is *forward* determinism:
   WD2 deliberately changed the mapping from the old MD5 selection to gain coherence, so
   seeds no longer reproduce pre-WD2 outputs — accepted because VD lands before the M6 big
   generation. See §3 (WD2, design principle).
4. **Web-ops scoring (VC):** does the interaction trajectory (which pages visited, what was
   clicked) count toward the score, or is interaction only the means to reach a final
   screenshot? **[OPEN]**
5. **Performance budget (VC):** acceptable to add several Chrome renders per episode? If
   not, C7 render caching must come first. **[OPEN]**

---

## 9. Code-anchor index

- **Render engine:** `vlm_asset_generators.py` — Chrome cmd `:149/231/738`,
  `render_chart_spec:117`, `render_raw_html:729`, `render_python_code:662`.
- **Scheduling:** `vlm_materializer.py` — `generate_vlm_assets:734`, per-platform
  pre-design `:791-890`, style-hash `:820-828`, info isolation `:835-844`, asset-detail
  `:382-513`, image-asset isolation `:795`.
- **LLM contracts:** `llm_client.py` — `compose_vlm_html:2704-2746`,
  `vlm_asset_detail:2601`, `generate_vlm_layout_spec:2651`,
  `generate_vlm_redundant_noise:2692`, `conflict_layer0c_assets_plan:2183`.
- **Platform normalization:** `conflict_llm_pipeline.py` — `_build_platform_vocab:2496`,
  `_sanitize_platform_id:2449`, `_normalize_and_prefix_assets:2539`,
  `_MIN_PLATFORM_VOCAB:2446`, blocklist `:2424`, default vocab `:2437`.
- **Specs:** `vlm_spec.py` — `VLMAssetSpec.platform_id:141`, `to_runtime_dict:307-315`,
  `VLMInputPolicy:62-75`; `conflict_vlm_spec.py` — `platforms:287-290`,
  `ConflictScenarioUpdate:60-80`.
- **Templates:** `templates/chart_template.html` (2 layouts + 10 visual styles),
  `templates/text_template.html`, `templates/libs/`; `chart_presets.py` (8 presets,
  no-LLM path only).
- **Conflict materializer visual bits:** `conflict_vlm_materializer.py` —
  `STYLE_PRESETS:1432-1459`, ≥2-platform soft WARN `:1461-1475`, anchor backfill
  `:3126-3225`.
- **Runtime / web:** `train_eval/agents.py` — `_multimodal_image_parts_from_task:81-138`,
  priming `:1296-1305`, `--multimodal-input-mode`; `core/observation.py` (text-only
  `:18-57`); `core/environment.py` — effect buckets `:102-124`, `_track_effects:1073-1244`,
  nested-effects passthrough `:1233-1244`; `backend/asset_backend.py`,
  `backend/mcp_facade.py`, `backend/mcp_server.py`, `backend/http_backend.py`,
  `backend/hybrid_backend.py`, `backend/multi_app_backend.py` (skill registration
  `:153-180`); `specs/agent_tools.v1.json`; `tools/agent_tool_spec.py`
  (`:195-206/:480-483`); `evaluation/checkers/effect_checker.py:126-171`.
- **Not to be confused:** `consistency_layer.py` reconciles task-scoring contradictions
  only — **not** visual/data consistency.

---

*This plan is intentionally scoped to the **visual/web surface**. Task mechanics,
difficulty, and conflict archetypes remain owned by
[`docs/conflict-expansion-plan.md`](./conflict-expansion-plan.md).*
