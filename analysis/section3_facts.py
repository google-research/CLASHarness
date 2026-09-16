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

"""Every number Section 3's prose cites, recomputed from the corpus.

Section 3 is prose, so unlike the body tables there is no generator whose
output can be diffed against the .tex. This script is the substitute: it
recomputes each figure the prose states, and `--check` asserts the values the
prose actually shipped. If a number in Section 3 and a number here disagree,
one of them is wrong and the assertion says which.

Two definitions here are deliberately aligned with what already ships in
Table~\\ref{tab:validity}, because prose that disagrees with an adjacent table
is worse than prose that is merely terse:

  deterministic checks = the exact complement of field_semantic_match.
      check_split_v8.py:58-59 defines DETERMINISTIC that way, and its "other"
      bucket (545) is any_field_equals (515) + within_window (30). So the
      count is 1623, not the 1593 that an earlier draft used after dropping
      within_window. A within_window check does carry an exact expected
      `value` -- it just adds a time bound on top -- so excluding it
      understated the deterministic surface and contradicted the table.

  checks per task = MEAN 6.27. The median is 6. An earlier comment in
      main.tex called 6.27 a median; 3760/600 is an average.

  gold length EXCLUDES the trailing DONE sentinel, via corpus.gold_length.
      An earlier draft counted it, which overstated every gold figure by
      exactly one step and, because slack is a ratio, put the slack
      denominator on a different footing from the budget numerator that is
      its numerator. Scored agents never emit DONE (0 occurrences in 600
      evaluation episodes), so it costs no budget.

Run:  python3 analysis/section3_facts.py            # print everything
      python3 analysis/section3_facts.py --check    # assert the shipped values

The extension-surface counts (3.6) and protocol families (3.7) are read from
the harness source tree, which ships here as clasharness/. CLASHARNESS_SRC
defaults to the repository root; point it elsewhere to check another checkout:

      python3 analysis/section3_facts.py --check

You do not have to remember to run this. analysis/results_facts.py calls
facts()/code_facts()/difficulty_facts() and asserts the SHIPPED dict below on
its own behalf, so `results_facts.py --check` covers Section 3 too. The direct
entry point above is kept because it prints all 98 values, which is what you
want while editing Section 3. SHIPPED stays here rather than moving there:
these numbers are RECOMPUTED from the corpus, and results_facts.py's design
note forbids recomputation in that file -- its values are parsed from
generator stdout, so that a statistic has exactly one derivation.
"""
import base64
import glob
import json
import os
import re
import statistics as st
import struct
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus  # noqa: E402  (path shim above must run first)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
# The two task corpora, fetched by tools/download_data.py (Corpus A is the v8
# main corpus, Corpus B the Sonnet-5 replication).
DATA_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
V8 = os.environ.get(
    "CLASHARNESS_TASKS_V8",
    os.path.join(DATA_ROOT, "multiagent_conflict_data_v8_Gemini_3_6_flash",
                 "tasks"))
SONNET = os.environ.get(
    "CLASHARNESS_TASKS_SONNET",
    os.path.join(DATA_ROOT, "multiagent_conflict_data_Gemini_Sonnet_5",
                 "tasks"))
# The harness source tree, which supplies the extension-surface counts in 3.6
# and the protocol families in 3.7. It is this repository by default; point
# CLASHARNESS_SRC elsewhere to check a different checkout. If it is not a
# directory the corpus figures still compute and --check reports the code-side
# values as unverified rather than silently passing on a subset.
CODE = os.environ.get("CLASHARNESS_SRC", REPO_ROOT)
BUDGET = 20

# Classes in clasharness/skills that subclass Skill but are not themselves a
# shipped skill: the protocol base, two backend adapters, and the docstring
# example that documents the extension decorator.
NOT_A_SKILL = {"StatefulProtocolSkill", "BackendSkillAdapter",
               "FileSkillAdapter", "MySkill"}


_LOADED = {}


def load(root):
    if root in _LOADED:
        return _LOADED[root]
    out = {}
    for p in sorted(glob.glob(f"{root}/*/specs.json")):
        d = json.load(open(p))
        out[d["task_id"]] = d
    _LOADED[root] = out
    return out


def facts():
    """Every cited quantity, as a flat dict."""
    v8 = load(V8)
    f = {"tasks": len(v8)}

    pkgs, seeds = defaultdict(int), defaultdict(int)
    for d in v8.values():
        slug = ((d.get("data") or {}).get("public") or {}).get("scenario_slug") or ""
        pkg = re.sub(r"_r\d+$", "", slug)
        pkgs[pkg] += 1
        seeds[re.sub(r"_v\d+$", "", pkg)] += 1
    f["seeds"] = len(seeds)
    f["packages"] = len(pkgs)
    f["rounds_per_seed"] = sorted(set(seeds.values()))
    f["rounds_per_package"] = sorted(set(pkgs.values()))

    checks, per_task = [], []
    for d in v8.values():
        cs = d["ground_truth"].get("evaluation_checks") or []
        checks += cs
        per_task.append(len(cs))
    f["checks"] = len(checks)
    f["checks_mean"] = round(st.mean(per_task), 2)
    f["checks_median"] = st.median(per_task)
    f["checks_min"], f["checks_max"] = min(per_task), max(per_task)

    cond = Counter(c.get("condition") for c in checks)
    for k in ("field_semantic_match", "field_contains", "any_field_equals",
              "within_window", "exists"):
        f[f"cond_{k}"] = cond.get(k, 0)
    f["deterministic"] = f["checks"] - f["cond_field_semantic_match"]

    gold = [corpus.gold_length(d["ground_truth"]) for d in v8.values()]
    f["gold_mean"] = round(st.mean(gold), 2)
    f["gold_median"] = st.median(gold)
    f["gold_min"], f["gold_max"] = min(gold), max(gold)

    # Deterministic checks all carry an exact expected string value.
    exact = [(c.get("expected") or {}).get("value") for c in checks
             if c.get("condition") != "field_semantic_match"]
    exact = [v for v in exact if isinstance(v, str) and v]
    f["exact_values"] = len(exact)
    f["exact_nodigit"] = sum(1 for v in exact if not any(ch.isdigit() for ch in v))
    f["exact_median_chars"] = st.median([len(v) for v in exact])

    crit = [(c.get("expected") or {}).get("criterion") for c in checks
            if c.get("condition") == "field_semantic_match"]
    crit = [c for c in crit if c]
    f["criteria"] = len(crit)
    f["criteria_distinct"] = len(set(crit))
    f["criteria_median_words"] = st.median([len(c.split()) for c in crit])
    f["criteria_channel"] = sum(
        1 for c in crit if re.search(r"#[A-Za-z0-9][-A-Za-z0-9_]*", c))

    # assets_accessed is the only effect_type that carries a role at all.
    aa, req_hist = 0, Counter()
    for d in v8.values():
        cs = [c for c in (d["ground_truth"].get("evaluation_checks") or [])
              if c.get("effect_type") == "assets_accessed"]
        aa += len(cs)
        req_hist[sum(1 for c in cs if c.get("role") == "required")] += 1
    f["asset_checks"] = aa
    f["checks_without_role"] = f["checks"] - aa
    f["asset_required_hist"] = dict(sorted(req_hist.items()))
    f["rounds_visual_required"] = sum(n for k, n in req_hist.items() if k >= 1)
    f["rounds_no_visual_score"] = f["tasks"] - f["rounds_visual_required"]

    # --- sec:consistency -------------------------------------------------
    # Role split of the 822 role-carrying checks, and the four markers the
    # fidelity-degradation path would leave behind. All four are absent: the
    # path never engaged in this corpus, which the prose reports as such.
    f["role_diagnostic"] = sum(
        1 for c in checks if c.get("role") == "diagnostic")
    f["role_required"] = sum(1 for c in checks if c.get("role") == "required")
    marks = Counter()
    for d in v8.values():
        blob = json.dumps(d)
        for t in ("fidelity_failed", "kept_with_failures", "repair_status",
                  "corroborating"):
            marks[t] += blob.count(t)
    for t, n in marks.items():
        f[f"mark_{t}"] = n

    # Auditor-3 (the chart-vs-workspace numeric auditor) reach. This is the
    # scope statement sec:consistency makes, so it must be measured rather
    # than asserted. The predicate below is copied verbatim from the auditor
    # itself (llm_auditor.py:72-77):
    #
    #     chart_data = {k: v for k, v in overrides.items()
    #                   if "chart" in k.lower()
    #                   and isinstance(v, dict)
    #                   and "__bytes_b64__" not in v}
    #
    # Every chart override in this corpus is a rendered PNG, so the b64 clause
    # empties the dict and the auditor never has input. Do NOT reimplement this
    # with a different notion of "chart" -- the number only means anything if
    # it uses the auditor's own test.
    #
    # NB: measure this against the RAW spec via load(), not corpus.load_specs(),
    # which returns a projection with no data/private key. Scanning the
    # projection silently yields 0 for everything and looks like a finding.
    ch_total = ch_dict_b64 = ch_nondict = ch_eligible = 0
    ch_rounds = 0
    for d in v8.values():
        ov = ((d.get("data") or {}).get("private") or {}).get(
            "initial_state_overrides") or {}
        elig = 0
        for k, v in ov.items():
            if "chart" not in k.lower():
                continue
            ch_total += 1
            if not isinstance(v, dict):
                ch_nondict += 1
            elif "__bytes_b64__" in v:
                ch_dict_b64 += 1
            else:
                elig += 1
        ch_eligible += elig
        if elig:
            ch_rounds += 1
    f["chart_overrides"] = ch_total
    f["chart_overrides_nondict"] = ch_nondict
    f["chart_overrides_rendered_png"] = ch_dict_b64
    f["chart_overrides_auditable"] = ch_eligible
    f["rounds_auditor3_active"] = ch_rounds

    # --- sec:assets ------------------------------------------------------
    # Shipped asset bytes, read out of data.private.initial_state_overrides.
    # Every asset key carries inline __bytes_b64__, so this is the corpus as
    # distributed rather than a manifest of files that may not exist. PNG
    # dimensions come from the IHDR in the first 24 decoded bytes; 3840x2160
    # is the configured 1920x1080 viewport at device-scale 2.0, so agreement
    # is the cheapest check that the renderer honoured the specification.
    dims, plats, per_asset, inline = Counter(), Counter(), [], 0
    plat_per_task = []
    for d in v8.values():
        iso = ((d.get("data") or {}).get("private") or {}).get(
            "initial_state_overrides") or {}
        keys = [k for k in iso if k.startswith("assets/")]
        per_asset.append(len(keys))
        seen = set()
        for k in keys:
            parts = k.split("/")
            if len(parts) >= 3:
                plats[parts[1]] += 1
                seen.add(parts[1])
            v = iso[k]
            if isinstance(v, dict) and "__bytes_b64__" in v:
                inline += 1
                head = base64.b64decode(v["__bytes_b64__"][:64] + "==")
                if head[:8] == b"\x89PNG\r\n\x1a\n":
                    dims[struct.unpack(">II", head[16:24])] += 1
        plat_per_task.append(len(seen))
    f["assets"] = sum(per_asset)
    f["assets_inline"] = inline
    f["assets_per_task_mean"] = round(st.mean(per_asset), 2)
    f["assets_per_task_median"] = st.median(per_asset)
    f["assets_per_task_min"], f["assets_per_task_max"] = (min(per_asset),
                                                          max(per_asset))
    f["assets_chrome"] = dims.get((3840, 2160), 0)
    f["assets_image_model"] = dims.get((1408, 768), 0)
    f["assets_placeholder"] = dims.get((960, 540), 0)
    f["asset_dim_kinds"] = len(dims)
    f["platform_surfaces"] = len(plats)
    f["platforms_per_task_mean"] = round(st.mean(plat_per_task), 2)

    # --- sec:diversity ---------------------------------------------------
    # Reuse is counted at ROUND granularity within a package: a value used by
    # two checks of the SAME round is target binding doing its job, not
    # repetition. A "repeat" is a value or criterion a previous round of the
    # same package already used.
    bypkg = defaultdict(list)
    for tid, d in v8.items():
        ev, cr = set(), set()
        for c in d["ground_truth"].get("evaluation_checks") or []:
            e = c.get("expected") or {}
            if c.get("condition") == "field_semantic_match":
                if isinstance(e.get("criterion"), str):
                    cr.add(e["criterion"])
            elif isinstance(e.get("value"), str) and e["value"].strip():
                ev.add(e["value"])
        bypkg[re.sub(r"_r\d+$", "", tid)].append((ev, cr))
    ev_slots = ev_rep = cr_slots = cr_rep = 0
    repeated, singular, dupe_rounds = [], [], 0
    for rs in bypkg.values():
        sv, sc = Counter(), Counter()
        for ev, cr in rs:
            sv.update(ev)
            sc.update(cr)
        ev_slots += sum(sv.values())
        ev_rep += sum(n - 1 for n in sv.values() if n > 1)
        cr_slots += sum(sc.values())
        cr_rep += sum(n - 1 for n in sc.values() if n > 1)
        repeated += [v for v, n in sv.items() if n > 1]
        singular += [v for v, n in sv.items() if n == 1]
        sigs = Counter((frozenset(ev), frozenset(cr)) for ev, cr in rs)
        dupe_rounds += sum(n - 1 for n in sigs.values() if n > 1)
    f["exact_round_slots"] = ev_slots
    f["exact_round_repeats"] = ev_rep
    f["criteria_round_slots"] = cr_slots
    f["criteria_round_repeats"] = cr_rep
    f["duplicate_rounds"] = dupe_rounds
    # What recurs is identifiers; what never recurs is readings. The prose
    # rests on this contrast, so guard both halves.
    # Both sides are counted as DISTINCT values, one vote each. An earlier
    # pass weighted the recurring side by multiplicity and the non-recurring
    # side by distinct value, which made the contrast look sharper than it is
    # (median 16 vs 44 rather than 30 vs 44). Keep the denominators equal.
    f["repeat_values"] = len(repeated)
    f["repeat_median_chars"] = st.median([len(v) for v in repeated])
    f["repeat_with_space"] = sum(1 for v in repeated if " " in v)
    f["repeat_with_digit"] = sum(
        1 for v in repeated if any(ch.isdigit() for ch in v))
    f["singular_values"] = len(singular)
    f["singular_median_chars"] = st.median([len(v) for v in singular])
    f["singular_with_space"] = sum(1 for v in singular if " " in v)
    f["singular_with_digit"] = sum(
        1 for v in singular if any(ch.isdigit() for ch in v))

    rt = Counter(((d.get("data") or {}).get("public") or {}).get("round_type")
                 for d in v8.values())
    f["round_type"] = dict(rt.most_common())
    # The dict above was printed but never asserted, so the one place the
    # paper spells the split out in words -- the scope limitation in sec:limits
    # -- was unguarded, and it is the only place the LABELS appear at all.
    # tab:coverage carries the bare counts "274/174/86/65/1" and is byte-diffed
    # against coverage_v8.py, so the counts alone were safe; which count
    # belongs to which round type was not. Both are pinned, as two strings, so
    # that a reordering fails rather than silently relabelling the corpus.
    f["round_type_counts"] = "/".join(str(n) for _, n in rt.most_common())
    f["round_type_labels"] = ",".join(k for k, _ in rt.most_common())
    f["anchors_nonempty"] = sum(
        1 for d in v8.values()
        if (d["ground_truth"].get("metadata") or {}).get("expected_anchor_ids"))

    slack = [BUDGET / g for g in gold if g]
    f["slack_mean"] = round(st.mean(slack), 2)
    f["slack_median"] = round(st.median(slack), 2)
    f["slack_min"] = round(min(slack), 2)
    # The paper quotes slack as the RANGE, "[1.11, 2.86]", in three places, and
    # only the lower end was ever computed here. 2.86 is 20/7 and so is pinned
    # by gold_min = 7 in arithmetic but not in the assertion set; a corpus edit
    # that shortened one gold trajectory would move the printed interval while
    # every shipped key still matched.
    f["slack_max"] = round(max(slack), 2)
    f["slack_tight"] = sum(1 for s in slack if s < 1.5)

    if os.path.isdir(SONNET):
        sn = load(SONNET)
        f["sonnet_tasks"] = len(sn)
        f["ids_identical"] = set(sn) == set(v8)
        common = sorted(set(sn) & set(v8))
        a = [corpus.gold_length(v8[t]["ground_truth"]) for t in common]
        b = [corpus.gold_length(sn[t]["ground_truth"]) for t in common]
        ma, mb = st.mean(a), st.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        f["gold_pearson_r"] = round(num / den, 3)
        f["sonnet_gold_mean"] = round(mb, 2)
        f["sonnet_longer"] = sum(1 for x, y in zip(a, b) if y > x)
        f["sonnet_equal"] = sum(1 for x, y in zip(a, b) if y == x)
        f["sonnet_shorter"] = sum(1 for x, y in zip(a, b) if y < x)
    return f


def code_facts():
    """Extension-surface counts, read off the source tree."""
    if not os.path.isdir(CODE):
        return {}

    def walkpy(root):
        o = []
        for dp, dn, fn in os.walk(root):
            if "__pycache__" in dp:
                continue
            o += [os.path.join(dp, x) for x in fn if x.endswith(".py")]
        return sorted(o)

    f, mods = {}, set()
    n = 0
    for p in walkpy(f"{CODE}/clasharness/scenario/plugins"):
        for l in open(p, errors="ignore"):
            # Decoration sites only. Counting bare occurrences also picks up
            # the `def` in registry.py and the docstring that cites it.
            if l.strip().startswith("@register_conflict_plugin"):
                n += 1
                mods.add(p)
    f["plugin_registrations"] = n
    f["plugin_modules"] = len(mods)

    cls, specs = {}, set()
    for p in walkpy(f"{CODE}/clasharness/skills"):
        t = open(p, errors="ignore").read()
        for m in re.finditer(r"class\s+(\w+)\(([^)]*)\)", t):
            if re.search(r"\b(Skill|StatefulProtocolSkill)\b", m.group(2)):
                cls[m.group(1)] = m.group(2)
        for m in re.finditer(
                r"""SkillSpec\(\s*\n?\s*name\s*=\s*["']([\w.\-]+)["']""", t):
            specs.add(m.group(1))
    f["skill_specs"] = len(specs)
    f["skill_impls"] = len([c for c in cls if c not in NOT_A_SKILL])
    f["protocol_families"] = sorted(
        c for c, b in cls.items() if "StatefulProtocolSkill" in b)

    # Categories and the effect-declaring subset. Only plugins with a
    # non-empty emits_effects participate in the derived command vocabulary
    # (_mechanic_command_sets); the challenge plugins are prompt-only.
    cats, with_effects = Counter(), 0
    src_names, effect_names = set(), set()
    for p in walkpy(f"{CODE}/clasharness/scenario/plugins"):
        t = open(p, errors="ignore").read()
        for m in re.finditer(r'category\s*=\s*"([a-z_]+)"', t):
            cats[m.group(1)] += 1
        # One registration per module (plugin_modules == plugin_registrations
        # == 23), so the module's first `name =` is its plugin id.
        nm = re.search(r'\bname\s*=\s*"([a-z0-9_]+)"', t)
        if nm:
            src_names.add(nm.group(1))
        for m in re.finditer(r"emits_effects\s*=\s*\(([^)]*)\)", t, re.S):
            if m.group(1).strip(" ,\n\"'"):
                with_effects += 1
                if nm:
                    effect_names.add(nm.group(1))
    f["plugin_categories"] = len(cats)
    f["plugins_with_effects"] = with_effects

    # sec:extensions points the reader at Table~\ref{tab:coverage} for "23
    # mechanics, 13 of which declare effects". That pointer is only honest if
    # the table's counts are the SAME SETS as the source-side ones and not
    # merely the same integers -- precisely the trap that
    # assets_image_model/platform_surfaces (both 117, unrelated) sets
    # elsewhere in this file. The table's 23 is seed-derived
    # (corpus.load_axis_map) and its "13 mechanic" is
    # design_axes.mechanic_plugins; both are asserted equal to the source-side
    # sets here, so regenerating either side cannot silently falsify the
    # cross-reference. It also settles what the factory's "mechanic" label
    # means -- it is exactly the effect-declaring subset, which the
    # coverage_v8 comment previously declined to guess at.
    seed_names = {p for v in corpus.load_axis_map()["seeds"].values()
                  for p in v["plugins"]}
    mech = set(json.load(open(corpus.DESIGN_AXES, encoding="utf-8"))
               ["mechanic_plugins"])
    f["plugin_names_match_corpus"] = src_names == seed_names
    f["mechanic_plugins_are_effect_declaring"] = mech == effect_names
    # "Adding one is a new module ... and a single import line."
    init = f"{CODE}/clasharness/scenario/plugins/__init__.py"
    f["plugin_import_lines"] = sum(
        1 for l in open(init, errors="ignore")
        if l.startswith("from clasharness.scenario.plugins.")
        and "import" in l and l.split(".")[3].split(" ")[0] not in
        ("base", "spec", "registry"))

    # sec:assets cites both of these.
    vs = open(f"{CODE}/clasharness/scenario/vlm_spec.py", errors="ignore").read()
    body = re.search(r"class VLMAssetSpec\b.*?(?=\nclass |\Z)", vs, re.S)
    f["asset_spec_fields"] = len(
        re.findall(r"^\s{4}([a-z_][a-z0-9_]*)\s*:", body.group(0), re.M))
    da = f"{CODE}/clasharness/scenario/design_archetypes.py"
    at = open(da, errors="ignore").read()
    f["design_archetypes"] = len(set(
        (a or b) for a, b in re.findall(
            r'^\s*"([a-z_]+)"\s*:\s*DesignArchetype|DesignArchetype\(\s*name="([a-z_]+)"',
            at, re.M)))

    seeds = sorted(glob.glob(
        f"{CODE}/clasharness/data/scenario_seeds/conflict_vlm_v1/*.json"))
    sets = [set(json.load(open(s))) for s in seeds]
    f["seed_files"] = len(seeds)
    f["seed_fields_required"] = len(set.intersection(*sets)) if sets else 0
    f["seed_fields_union"] = len(set().union(*sets)) if sets else 0

    # --- sec:difficulty --------------------------------------------------
    # The seeds split into those that declare the optional difficulty block
    # and those that omit it. NOTE THE CONFOUND, which the prose states: the
    # declaring seeds declare the block as a GROUP, so a difference between
    # the two arms cannot be attributed to any single field. The one effect
    # that IS attributable is the visual one, because force_vision_locus is
    # the only field in the block that touches the visual channel.
    common = set.intersection(*sets) if sets else set()
    declares = {os.path.basename(s)[:-5] for s in seeds
                if "force_vision_locus" in set(json.load(open(s))) - common}
    f["seeds_declaring_block"] = len(declares)
    f["seeds_omitting_block"] = len(seeds) - len(declares)
    return f


def difficulty_facts():
    """The sec:difficulty opt-in split, joining seeds to the rounds they grew.

    A task id is ``conflict_vlm_<seed stem>_v<NN>_r<NN>``, so the seed a round
    came from is recovered by longest-prefix match against the seed stems --
    longest, because ``multi_device_pairing`` and ``multi_party_access_permit``
    share a prefix with nothing but each other only under a shortest match.

    The two arms are NOT a controlled experiment and the prose says so: the
    fourteen seeds declare the whole optional block together. The one claim
    that survives the confound is the visual one, because force_vision_locus
    is the only field in the block whose meaning reaches the image channel.
    """
    if not os.path.isdir(CODE):
        return {}
    seeds = sorted(glob.glob(
        f"{CODE}/clasharness/data/scenario_seeds/conflict_vlm_v1/*.json"))
    if not seeds:
        return {}
    sets = {os.path.basename(s)[:-5]: set(json.load(open(s))) for s in seeds}
    common = set.intersection(*sets.values())
    declares = {k for k, v in sets.items()
                if "force_vision_locus" in v - common}

    v8 = load(V8)
    arms = {"optin": [], "default": []}
    for tid, d in v8.items():
        body = re.sub(r"^conflict_vlm_", "", tid)
        stem = max((s for s in sets if body.startswith(s)), key=len,
                   default=None)
        if stem is None:
            continue
        arms["optin" if stem in declares else "default"].append(d)

    f = {}
    for arm, ds in arms.items():
        gold = [corpus.gold_length(d["ground_truth"]) for d in ds]
        req = sum(
            1 for d in ds
            if any(c.get("effect_type") == "assets_accessed"
                   and c.get("role") == "required"
                   for c in (d["ground_truth"].get("evaluation_checks") or [])))
        f[f"{arm}_rounds"] = len(ds)
        f[f"{arm}_gold_min"] = min(gold)
        f[f"{arm}_gold_median"] = st.median(gold)
        f[f"{arm}_visual_required"] = req
    f["optin_visual_pct"] = round(
        100 * f["optin_visual_required"] / f["optin_rounds"], 1)
    f["optin_degraded"] = f["optin_rounds"] - f["optin_visual_required"]
    return f


# The values Section 3 actually states. Keep in lockstep with main.tex.
SHIPPED = {
    "tasks": 600, "seeds": 20, "packages": 60,
    "checks": 3760, "checks_mean": 6.27, "checks_median": 6,
    "cond_field_semantic_match": 2137, "cond_field_contains": 1078,
    "cond_any_field_equals": 515, "cond_within_window": 30, "cond_exists": 0,
    "deterministic": 1623,
    "gold_mean": 10.28, "gold_median": 10, "gold_min": 7, "gold_max": 18,
    "exact_values": 1623, "exact_nodigit": 507, "exact_median_chars": 28,
    "criteria": 2137, "criteria_distinct": 2086, "criteria_median_words": 19,
    "criteria_channel": 1172,
    "asset_checks": 822, "checks_without_role": 2938,
    "rounds_visual_required": 346, "rounds_no_visual_score": 254,
    "anchors_nonempty": 600,
    # sec:consistency -- the role split of the 822, and the four markers the
    # fidelity-degradation path would leave. All four must stay at zero for
    # the prose sentence "it never engaged at all" to hold.
    "role_diagnostic": 476, "role_required": 346,
    "mark_fidelity_failed": 0, "mark_kept_with_failures": 0,
    "mark_repair_status": 0, "mark_corroborating": 0,
    # Auditor-3 reach, using the auditor's own predicate. All 602 chart
    # overrides are rendered PNGs, so none is auditable and the auditor has
    # input on 0 of 600 rounds. This backs the scope disclosure in
    # sec:consistency, which must stay a SCOPE statement: the auditor could
    # only ever have policed agreement among non-decisive numbers, and nothing
    # non-decisive is graded.
    "chart_overrides": 602, "chart_overrides_nondict": 0,
    "chart_overrides_rendered_png": 602, "chart_overrides_auditable": 0,
    "rounds_auditor3_active": 0,
    # sec:assets
    # CAUTION: assets_image_model and platform_surfaces are both 117 by
    # coincidence -- one counts 1408x768 assets, the other counts distinct
    # platform directories. They are unrelated; do not collapse them.
    "assets": 7970, "assets_inline": 7970, "assets_per_task_mean": 13.28,
    "assets_per_task_median": 13, "assets_per_task_min": 1,
    "assets_per_task_max": 34, "asset_dim_kinds": 3,
    "assets_chrome": 7843, "assets_image_model": 117,
    "assets_placeholder": 10, "platform_surfaces": 117,
    "platforms_per_task_mean": 2.75,
    # sec:diversity
    "exact_round_slots": 1504, "exact_round_repeats": 486,
    "criteria_round_slots": 2137, "criteria_round_repeats": 35,
    "duplicate_rounds": 0,
    "repeat_values": 168, "repeat_median_chars": 30, "repeat_with_space": 0,
    "singular_values": 850, "singular_median_chars": 44,
    "singular_with_space": 77, "singular_with_digit": 772,
    # "budget slack (1.11--2.86)" / "$b / \text{gold} \in [1.11, 2.86]$".
    "slack_mean": 2.02, "slack_min": 1.11, "slack_max": 2.86,
    "slack_tight": 49,
    # sec:limits -- "274 investigation, 174 verification, 86 action, 65
    # decision, and 1 reconciliation round", and the counts half of it is also
    # the tab:coverage cell that coverage_v8.py emits.
    "round_type_counts": "274/174/86/65/1",
    "round_type_labels":
        "investigation,verification,action,decision,reconciliation",
    "ids_identical": True, "gold_pearson_r": 0.186, "sonnet_gold_mean": 12.48,
    "sonnet_longer": 430, "sonnet_equal": 72, "sonnet_shorter": 98,
    "plugin_registrations": 23, "plugin_modules": 23,
    "plugin_categories": 7, "plugins_with_effects": 13,
    "plugin_import_lines": 23,
    # Set identities, not counts. These guard the Table~\ref{tab:coverage}
    # cross-reference in sec:extensions; see the comment at their definition.
    "plugin_names_match_corpus": True,
    "mechanic_plugins_are_effect_declaring": True,
    "skill_impls": 19, "skill_specs": 22,
    "seed_files": 20, "seed_fields_required": 16, "seed_fields_union": 24,
    "asset_spec_fields": 27, "design_archetypes": 15,
    # sec:difficulty -- the opt-in split. Confounded by construction (the 14
    # declare the whole optional block together); the prose says so.
    "seeds_declaring_block": 14, "seeds_omitting_block": 6,
    "optin_rounds": 420, "optin_gold_min": 8, "optin_gold_median": 10,
    "optin_visual_required": 346, "optin_visual_pct": 82.4,
    "optin_degraded": 74,
    "default_rounds": 180, "default_gold_min": 7, "default_gold_median": 8,
    "default_visual_required": 0,
}


def main():
    f = facts()
    f.update(code_facts())
    f.update(difficulty_facts())
    for k, v in f.items():
        print(f"  {k:28} {v}")
    print(f"\n  round_type sums to {sum(f['round_type'].values())}")
    print(f"  every deterministic check has an exact value: "
          f"{f['exact_values'] == f['deterministic']}")

    if "--check" in sys.argv:
        bad = []
        for k, want in SHIPPED.items():
            if k not in f:
                bad.append(f"{k}: not computed (source tree unreachable?)")
            elif f[k] != want:
                bad.append(f"{k}: prose says {want}, corpus says {f[k]}")
        if bad:
            raise SystemExit("SECTION 3 DRIFT\n  " + "\n  ".join(bad))
        print(f"\n  OK: all {len(SHIPPED)} values cited in Section 3 reproduce.")


if __name__ == "__main__":
    main()
