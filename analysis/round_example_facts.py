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

"""Every claim the three round-example exhibits make, checked against the corpus.

Appendix B gained a worked round and an asset gallery, and with them a class of
claim the paper did not previously make: quoted strings. Tables and figures
elsewhere cite numbers that a generator recomputes, but "the dashboard reads
UNRECONCILED (2/3 approver scopes mismatched)" is a sentence about pixels, and
nothing in the repo could tell whether it was still true. This module makes it
checkable -- it re-reads the shipped corpus and asserts, literally, every string
and count that Figures A/B/C, Table~\\ref{tab:round-checks} and the
sec:app-assets paragraph put on the page.

WHAT IT CHECKS AND WHY EACH ONE EARNS ITS PLACE

  Round anatomy    The round is described as having seven visible assets, one
                   required gate, eight checks and a fifteen-step gold. Each is
                   read back out of specs.json rather than restated.

  Evidence split   The figure's whole argument is that this round's decisive
                   strings live in two DIFFERENT renders -- so the checker
                   asserts the split in both directions: the reconciliation
                   verdict is in the gated matrix and NOT in the base one, the
                   passcode is in the base one and NOT in the gated one. An
                   asset regeneration that merged them would quietly make the
                   two-panel figure pointless, and only a negative assertion
                   catches that.

  Exclusivity      Both strings, and the stale-cache round's v*, are asserted
                   ABSENT from every surface the model can read as text: the
                   instruction, the workspace tree, the background messages and
                   the mock sub-agent replies. This is the sec:exclusivity claim
                   restated on a concrete round. specs.json itself is excluded
                   from that scan on purpose -- it holds ground_truth, which is
                   the grader's copy and is never shown to the model.

  Withheld assets  Figure C shows two of its round's eight visible assets and
                   the caption explains why the other six are out. Two of those
                   reasons are load-bearing -- an OCR buffer that prints BOTH
                   totals in one card, and a declared resolved_value_locus that
                   is a different render from the gated one -- so both are
                   asserted rather than described. A corpus regen that fixed
                   either would make a published caveat false in our own favour,
                   which is the direction of error worth catching.

  Gallery          kind and visual_style for all six tiles, because the tile
                   captions name them. preset_id is deliberately NOT captioned
                   and so is NOT checked: the renderer does not honour it (the
                   tile declared preset_gantt_chart draws a bar chart, the one
                   declared preset_pie_chart draws horizontal bars), and a
                   caption naming presets would be contradicted by its own
                   picture.

  Archetypes       The 18/16/14 counts and the kind histogram in sec:app-assets.
                   18 and 16 come from the harness source (design_archetypes.py,
                   via CLASHARNESS_SRC) since they are properties of the library,
                   not of the corpus; 14 and the histogram come from scanning
                   all 60 asset_manifest.json files.

CASE-SENSITIVITY IS NOT UNIFORM, ON PURPOSE. Several labels are title-case in
the HTML and uppercase in the render because a CSS class transforms them --
"Security Telemetry Passcode" ships, SECURITY TELEMETRY PASSCODE is what the
figure shows. Those are matched case-insensitively. Identifier-like values
(SEC-PASS-8842-BP, UNRECONCILED, VERIFIED_CACHE_VALIDATED, 725.46 EUR) are
matched exactly, because for those the casing IS the value.

FIGURE C'S TIMES COME FROM THE FILENAMES, NOT THE PIXELS. Panel (a) prints
"Snapshot ID: 20241012_084210" on its own face, but panel (b) does not print
its own stamp anywhere -- the only place 20241012_084610 exists is the asset
identifier. The renders also carry unrelated clock times (a console log at
10:14, an ISO stamp of 2024-10-24 on the scan, a stay date of 2024-10-10 on the
receipt), so a bare timestamp regex over the page would have picked one of
those. The "four minutes apart" claim is therefore derived from the two asset
IDs and the caption attributes it to them; both the positive (a prints its id)
and the negative (b does not) are asserted so the attribution stays honest.

Run:  python3 analysis/round_example_facts.py   # print every claim, and exit
                                               # non-zero if any has drifted

There is no separate --check mode: printing and asserting are one pass, so there
is no way to read these claims without also checking them. The tally and the
mismatch list additionally cross into results_facts.py --check via SECTIONS.

Needs the corpus (CLASHARNESS_SCENARIOS) and the harness source (CLASHARNESS_SRC).
"""
import ast
import collections
import datetime
import glob
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

# The generated scenario packages, fetched by tools/download_data.py; and the
# harness itself, which is this repository unless CLASHARNESS_SRC redirects it.
SCENARIOS = os.environ.get(
    "CLASHARNESS_SCENARIOS",
    os.path.join(DATA_ROOT, "scenarios", "conflict_vlm_v8_3.6_flash"))
SRC = os.environ.get("CLASHARNESS_SRC", REPO_ROOT)

GATE_PKG = "access_request_approval_v01"
GATE_ROUND = "conflict_vlm_access_request_approval_v01_r01"
GATE_ASSET = "iam_console_access_scope_matrix_r01"
BASE_ASSET = "iam_console_access_scope_matrix"

CACHE_PKG = "expense_reimbursement_conflict_v03"
CACHE_ROUND = "conflict_vlm_expense_reimbursement_conflict_v03_r03"
CACHE_ASSET = "expense_portal_legacy_cache_status_20241012_084210"   # panel (a)
SCAN_ASSET = "expense_portal_receipt_photo_20241012_084610"          # panel (b)
# Shown in neither panel, and the caption says so; both are asserted below.
DECOY_ASSET = "expense_portal_ocr_stream_buffer_20241012_084610"
LOCUS_ASSET = "expense_portal_receipt_photo"

VERDICT = "2/3 approver scopes mismatched"
PASSCODE = "SEC-PASS-8842-BP"

CACHE_VSTAR = "725.46 EUR"
CACHE_DECOY = "850.00 EUR"
# The five line items the figure invites the reader to add up, and the VAT rate
# that turns their sum into v*. Asserted as arithmetic, not as a literal.
RECEIPT_LINES = (260.00, 260.00, 45.00, 95.00, 18.00)
RECEIPT_VAT = 0.07

# out-of-figure assets that disagree with the two panels about one string; the
# generator's docstring explains why they are excluded, and this pins the
# disagreement so that a corpus regen which FIXED it would tell us (the note
# could then be dropped) rather than leaving a stale caveat in the source.
DISAGREEING = {
    "audit_portal_summary_card_20260330_083200": "billing-analytics",
    "security_dashboard_telemetry_20260330_083200": "analytics-prod",
}

GALLERY = [
    ("distributed_lock_shared_resource_v01",
     "warehouse_ops_portal_bay_pallet_photo", "natural_image", None),
    ("access_request_approval_v01",
     "audit_portal_access_policy_doc", "rich_text", "terminal_tui"),
    ("incident_commander_selection_v01",
     "network_manager_routing_topology_diagram", "diagram", "neo_brutalism"),
    ("clinical_trial_dosage_conflict_v01",
     "pv_dashboard_remapping_telemetry_20231112_103000", "chart", "bloomberg_terminal"),
    ("incident_commander_selection_v03",
     "ops_console_secondary_status_summary_20260330_105500", "chart", "github_dark"),
    ("payment_chargeback_dispute_v01",
     "risk_dashboard_proxy_ip_geolocation_chart_20231024_143500", "chart", "modern_clean"),
]

UNSAMPLED = ("discord_chat", "spotify_media")


# --- readers ----------------------------------------------------------------

def _spec(pkg, round_id):
    p = os.path.join(SCENARIOS, pkg, "tasks", round_id, "specs.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing round spec {p}\nset CLASHARNESS_SCENARIOS")
    return json.load(open(p, encoding="utf-8"))


def _render_text(pkg, asset_id):
    """Visible text of an asset's render, tags stripped, whitespace collapsed."""
    p = os.path.join(SCENARIOS, pkg, "assets", f"{asset_id}.render.html")
    if not os.path.exists(p):
        raise SystemExit(f"missing render {p}")
    raw = open(p, encoding="utf-8").read()
    t = re.sub(r"<(script|style)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def _manifest(pkg):
    return json.load(open(os.path.join(SCENARIOS, pkg, "asset_manifest.json"),
                          encoding="utf-8"))["assets"]


def _style_of(asset):
    return ((asset.get("custom_overrides") or {}).get("styling") or {}).get("visual_style")


def _text_surfaces(pkg, round_id):
    """Every string the model can reach WITHOUT opening an asset.

    ground_truth lives in the same specs.json and holds both decisive values,
    so this walks the named model-facing fields instead of the whole file.

    initial_state_overrides needs filtering rather than dumping. It is keyed by
    destination path, and six of the temporal round's nineteen entries are the
    rendered PNGs carried inline as base64 -- so a naive json.dumps of the whole
    mapping searches the IMAGE BYTES for the decisive value and finds it, which
    is exactly backwards: "126" turning up in the base64 of a PNG is a property
    of the byte stream, not a text leak, and the first version of this checker
    failed on precisely that.

    The filter is on the destination path alone. Filtering on "is it a str"
    instead was the second wrong answer: it also drops asset_manifest.json,
    emails.json and tasks.json, which are dict-valued but ARE readable state,
    and dropping readable state from an exclusivity scan is the one direction
    of error that makes the scan lie in the paper's favour.
    """
    d = _spec(pkg, round_id)
    priv = (d.get("data") or {}).get("private") or {}
    out = [d.get("instruction") or "", d.get("canonical_instruction") or ""]
    out += [json.dumps(d.get("visible_constraints") or []),
            json.dumps(d.get("decision_requirements") or []),
            json.dumps((d.get("data") or {}).get("public") or {})]
    out.append(json.dumps(priv.get("background_events") or []))
    out.append(json.dumps(priv.get("mock_subagents") or []))
    for dest, body in (priv.get("initial_state_overrides") or {}).items():
        if not dest.startswith("assets/"):
            out.append(body if isinstance(body, str) else json.dumps(body))
    for f in sorted(glob.glob(os.path.join(SCENARIOS, pkg, "workspace", "**", "*"),
                              recursive=True)):
        if os.path.isfile(f):
            try:
                out.append(open(f, encoding="utf-8", errors="replace").read())
            except OSError:
                pass
    return "\n".join(out)


def _archetype_library():
    """(defined, non-legacy) from the harness source, parsed not imported."""
    p = os.path.join(SRC, "clasharness", "scenario", "design_archetypes.py")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}\nset CLASHARNESS_SRC to a clash-harness root")
    tree = ast.parse(open(p, encoding="utf-8").read())
    names, legacy = [], []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None)
                == "DesignArchetype"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if "name" not in kw:
            continue
        names.append(ast.literal_eval(kw["name"]))
        legacy.append(bool(ast.literal_eval(kw["legacy"])) if "legacy" in kw else False)
    return names, [n for n, lg in zip(names, legacy) if not lg]


def _corpus_assets():
    """Every asset in all 60 packages, as (kind, style, preset) triples."""
    rows = []
    for man in sorted(glob.glob(os.path.join(SCENARIOS, "*", "asset_manifest.json"))):
        for a in json.load(open(man, encoding="utf-8"))["assets"]:
            rows.append((a.get("kind"), _style_of(a), a.get("preset_id")))
    return rows


# --- the checks -------------------------------------------------------------

def _compare():
    seen, bad = {}, []

    def check(key, got, want):
        seen[key] = got
        if got != want:
            bad.append(f"{key}: corpus says {got!r}, paper says {want!r}")

    def has(key, hay, needle, want=True, fold=False):
        h, n = (hay.lower(), needle.lower()) if fold else (hay, needle)
        check(key, (n in h), want)

    # -- Figure A / Table: round anatomy ------------------------------------
    d = _spec(GATE_PKG, GATE_ROUND)
    priv = d["data"]["private"]
    gt = d["ground_truth"]
    checks = gt["evaluation_checks"]

    check("gate.round_type", d["data"]["public"]["round_type"], "investigation")
    check("gate.difficulty", d["difficulty"], 3)
    check("gate.visible_assets", len(priv["visible_asset_ids"]), 7)
    check("gate.asset_visible", GATE_ASSET in priv["visible_asset_ids"], True)
    check("gate.base_visible", BASE_ASSET in priv["visible_asset_ids"], True)
    check("gate.n_checks", len(checks), 8)

    cmds = gt["solution_commands"]
    check("gate.gold_sentinel", str(cmds[-1]).strip(), "DONE")
    check("gate.gold_steps", len(cmds) - 1, 15)

    conds = collections.Counter(c.get("condition") for c in checks)
    check("gate.n_field_contains", conds["field_contains"], 1)
    check("gate.n_semantic", conds["field_semantic_match"], 5)
    check("gate.n_any_field_equals", conds["any_field_equals"], 2)

    gates = [c for c in checks if c.get("role") == "required"]
    check("gate.n_required_role", len(gates), 1)
    check("gate.gate_effect", gates[0]["effect_type"], "assets_accessed")
    check("gate.gate_target", gates[0]["expected"]["value"], GATE_ASSET)
    # every other check carries no role, i.e. is scored and is not the gate
    check("gate.n_roleless", sum(1 for c in checks if c.get("role") is None), 7)

    effects = [c["effect_type"] for c in checks]
    check("gate.effect_order", tuple(effects),
          ("assets_accessed", "messages_sent", "messages_sent", "approval_recorded",
           "approval_granted", "files_created", "files_created", "messages_sent"))

    crit = " || ".join(json.dumps(c["expected"]) for c in checks)
    has("gate.verdict_in_checks", crit, VERDICT)
    has("gate.passcode_in_checks", crit, PASSCODE)
    has("gate.permit_in_checks", crit, "PERMIT-2026-BP-9982")
    has("gate.report_in_checks", crit, "/reports/access_grant_decision.md")

    # the manager's claim the round is built to contradict
    quinn = [e for e in priv["background_events"]
             if isinstance(e, dict)
             and (e.get("message") or {}).get("message_id") == "msg_107"]
    check("gate.quinn_events", len(quinn), 1)
    check("gate.quinn_sender", quinn[0]["message"]["sender"], "Quinn_Engineering_Manager")
    has("gate.quinn_claim", quinn[0]["message"]["text"],
        "All three required stakeholders show a completed status icon")
    has("gate.instruction_claim", d["instruction"],
        "summary indicators in the portal display sign-offs from all required authorities")

    # -- the evidence split, asserted in both directions ---------------------
    gate_txt, base_txt = _render_text(GATE_PKG, GATE_ASSET), _render_text(GATE_PKG, BASE_ASSET)
    has("splitA.verdict_in_gated", gate_txt, VERDICT, want=True)
    has("splitA.verdict_in_base", base_txt, VERDICT, want=False)
    has("splitA.passcode_in_base", base_txt, PASSCODE, want=True)
    has("splitA.passcode_in_gated", gate_txt, PASSCODE, want=False)

    # panel (a) strings
    has("panelA.unreconciled", gate_txt, "UNRECONCILED")
    has("panelA.title", gate_txt, "Request Reconciliation: REQ-2026-0842")
    has("panelA.anchor", gate_txt, "access_scope_matrix_r01")
    has("panelA.narrower", gate_txt, "MISMATCH: Narrower")
    has("panelA.target_mismatch", gate_txt, "TARGET-MISMATCH (billing-archive)")
    has("panelA.matched", gate_txt, "WRITE (Matched)")
    # The caption says three of eight rows belong to the request under review
    # and two of those three are mismatched, which is where "2/3" comes from.
    # Counted off the table markup, not off the flattened text: REQ-2026-0842
    # also appears in the panel title and the anchor line.
    gate_html = open(os.path.join(SCENARIOS, GATE_PKG, "assets",
                                  f"{GATE_ASSET}.render.html"), encoding="utf-8").read()
    body = re.search(r"<tbody[^>]*>(.*?)</tbody>", gate_html, re.S | re.I).group(1)
    trs = re.split(r"<tr\b", body, flags=re.I)[1:]
    check("panelA.table_rows", len(trs), 8)
    subject = [r for r in trs if "REQ-2026-0842" in r]
    check("panelA.subject_rows", len(subject), 3)
    check("panelA.subject_mismatched",
          sum(1 for r in subject if "MISMATCH" in r), 2)
    check("panelA.distractor_rows", len(trs) - len(subject), 5)
    # panel (b) strings -- these two are CSS-uppercased, hence fold=True
    has("panelB.label_passcode", base_txt, "Security Telemetry Passcode", fold=True)
    has("panelB.label_state", base_txt, "Reconciliation State", fold=True)
    has("panelB.passcode", base_txt, PASSCODE)

    # the two panels agree on the string the excluded assets disagree about
    has("splitA.archive_in_gated", gate_txt, "billing-archive")
    has("splitA.archive_in_base", base_txt, "billing-archive")
    for aid, other in DISAGREEING.items():
        has(f"splitA.disagree.{aid}", _render_text(GATE_PKG, aid), other)

    # -- exclusivity ---------------------------------------------------------
    surf = _text_surfaces(GATE_PKG, GATE_ROUND)
    has("excl.verdict_in_text", surf, VERDICT, want=False)
    has("excl.passcode_in_text", surf, PASSCODE, want=False)

    # -- Figure C: the stale-cache round -------------------------------------
    cd = _spec(CACHE_PKG, CACHE_ROUND)
    cpriv = cd["data"]["private"]
    cchecks = cd["ground_truth"]["evaluation_checks"]

    check("cache.round_type", cd["data"]["public"]["round_type"], "investigation")
    check("cache.difficulty", cd["difficulty"], 3)
    check("cache.visible_assets", len(cpriv["visible_asset_ids"]), 8)
    check("cache.n_checks", len(cchecks), 5)
    ccmds = cd["ground_truth"]["solution_commands"]
    check("cache.gold_sentinel", str(ccmds[-1]).strip(), "DONE")
    check("cache.gold_steps", len(ccmds) - 1, 10)
    check("cache.anchor_ids", list(cpriv["expected_anchor_ids"]),
          ["EVIDENCE:RECEIPT_PHOTO_R3"])

    # Unlike the round this figure used to show, THIS one is gated: it belongs
    # to the 346-round stratum the prose puts it in, and the gate is on the
    # asset panel (b) is cut from. Both halves are asserted.
    cgates = [c for c in cchecks if c.get("role") == "required"]
    check("cache.n_required_role", len(cgates), 1)
    check("cache.gate_effect", cgates[0]["effect_type"], "assets_accessed")
    check("cache.gate_target", cgates[0]["expected"]["value"], SCAN_ASSET)
    check("cache.n_roleless", sum(1 for c in cchecks if c.get("role") is None), 4)

    # v* is graded by string equality, not by a judge -- the prose says so.
    det = [c for c in cchecks if c.get("condition") == "field_contains"
           and c["effect_type"] == "files_created"]
    check("cache.n_deterministic_value_checks", len(det), 1)
    check("cache.deterministic_value", det[0]["expected"]["value"], CACHE_VSTAR)
    # ...and the gold trajectory opens the gated asset first, as the prose says.
    check("cache.gold_opens_scan_first",
          str(ccmds[0]).strip(), f"asset open --id {SCAN_ASSET}")

    # -- the two panels ------------------------------------------------------
    cache_txt, scan_txt = _render_text(CACHE_PKG, CACHE_ASSET), _render_text(CACHE_PKG, SCAN_ASSET)
    # panel (a): the confident wrong number, and NOT the right one. The negative
    # is the load-bearing one -- it is what lets the caption say a reader of
    # panel (a) alone has no way to reach v*.
    has("panelC_a.title", cache_txt, "OCR Parser Buffer Telemetry")
    has("panelC_a.snapshot_id", cache_txt, "20241012_084210")
    has("panelC_a.batch", cache_txt, "SCAN_EXP8891_LEGACY")
    has("panelC_a.engine", cache_txt, "OCR-v1.9-legacy")
    has("panelC_a.status_label", cache_txt, "Engine Status", fold=True)
    has("panelC_a.status", cache_txt, "VERIFIED_CACHE_VALIDATED")
    has("panelC_a.extraction_label", cache_txt, "Cached Extraction", fold=True)
    has("panelC_a.decoy", cache_txt, CACHE_DECOY)
    has("panelC_a.vstar_present", cache_txt, "725.46", want=False)

    # panel (b): the ledger, every line the figure shows.
    has("panelC_b.title", scan_txt, "Verified Paper Receipt breakdown")
    has("panelC_b.folio", scan_txt, "GHR-99201")
    has("panelC_b.anchor", scan_txt, "EVIDENCE:RECEIPT_PHOTO_R3")
    has("panelC_b.hotel", scan_txt, "Grand Hotel Berlin")
    has("panelC_b.date", scan_txt, "2024-10-10")
    has("panelC_b.ledger_label", scan_txt, "Receipt Line-Item Ledger", fold=True)
    for label, amount in (("Room Charges (Night 1)", "260.00 EUR"),
                          ("Room Charges (Night 2)", "260.00 EUR"),
                          ("Breakfast & Dining", "45.00 EUR"),
                          ("Dinner Services", "95.00 EUR"),
                          ("City Tourism Tax", "18.00 EUR")):
        has(f"panelC_b.line.{label}", scan_txt, label)
        has(f"panelC_b.amt.{amount}", scan_txt, amount)
    has("panelC_b.subtotal", scan_txt, "678.00 EUR")
    has("panelC_b.vat_label", scan_txt, "Tax (VAT 7%)")
    has("panelC_b.vat", scan_txt, "47.46 EUR")
    has("panelC_b.total_label", scan_txt, "Grand Total", fold=True)
    has("panelC_b.total", scan_txt, CACHE_VSTAR)

    # The caption invites the reader to add the panel up; if the corpus ever
    # re-renders with different line items that invitation becomes a trap, so
    # the arithmetic is checked rather than transcribed.
    check("cache.lines_sum", f"{sum(RECEIPT_LINES):.2f}", "678.00")
    check("cache.vat_applied",
          f"{sum(RECEIPT_LINES) * (1 + RECEIPT_VAT):.2f}", "725.46")

    # -- the two assets the caption says are withheld, and why ---------------
    # If a corpus regen ever made these agree with panel (b), the caption's
    # caveats would become false in the paper's favour -- so both are pinned.
    decoy_txt = _render_text(CACHE_PKG, DECOY_ASSET)
    check("cache.decoy_visible", DECOY_ASSET in cpriv["visible_asset_ids"], True)
    has("cache.decoy_has_vstar", decoy_txt, CACHE_VSTAR)
    has("cache.decoy_has_decoy", decoy_txt, CACHE_DECOY)
    has("cache.decoy_stale_label", decoy_txt, "Stale Cached Total")
    has("cache.decoy_verified_label", decoy_txt, "Verified Physical Receipt Total")

    # The declared locus is a DIFFERENT asset from the gated one -- the caption
    # calls this a corpus inconsistency, so it must actually hold.
    check("cache.locus_visible", LOCUS_ASSET in cpriv["visible_asset_ids"], True)
    check("cache.locus_is_not_gate", LOCUS_ASSET != SCAN_ASSET, True)
    # ...and it is the natural photograph, which is image-generated and so has
    # no render.html to read; that is why the figure cuts the rendered scan.
    check("cache.locus_render_exists",
          os.path.exists(os.path.join(SCENARIOS, CACHE_PKG, "assets",
                                      f"{LOCUS_ASSET}.render.html")), False)
    check("cache.locus_kind",
          next(a.get("kind") for a in _manifest(CACHE_PKG) if a["id"] == LOCUS_ASSET),
          "natural_image")

    # -- exclusivity ---------------------------------------------------------
    ciso = cpriv["initial_state_overrides"]
    check("cache.iso_entries", len(ciso), 16)
    check("cache.iso_dropped", sum(1 for k in ciso if k.startswith("assets/")), 6)
    check("cache.iso_dropped_all_encoded",
          all(isinstance(v, dict) for k, v in ciso.items() if k.startswith("assets/")), True)
    check("cache.iso_kept_nonstr",
          sorted(k for k, v in ciso.items()
                 if not k.startswith("assets/") and not isinstance(v, str)),
          ["asset_manifest.json", "emails.json", "tasks.json"])
    # The prose enumerates these ten by category, so the enumeration is pinned:
    # an entry that appears later must be named there rather than silently
    # folded into a count that still reads "ten".
    kept = sorted(k for k in ciso if not k.startswith("assets/"))
    check("cache.iso_kept", len(kept), 10)
    check("cache.iso_kept_workspace",
          sum(1 for k in kept if k.startswith("workspace/")), 5)
    check("cache.iso_kept_sessions",
          sum(1 for k in kept if k.startswith("sessions/")), 2)
    check("cache.iso_kept_toplevel",
          [k for k in kept if "/" not in k],
          ["asset_manifest.json", "emails.json", "tasks.json"])

    csurf = _text_surfaces(CACHE_PKG, CACHE_ROUND)
    # The paper quotes this length, so it is pinned: a surface that silently
    # shrank would weaken the exclusivity claim without anything failing.
    check("cache.text_surface_chars", len(csurf), 25526)
    # v* absent, and so is each half it is built from -- a model cannot even
    # reconstruct it by arithmetic from text.
    has("cache.vstar_in_text", csurf, "725.46", want=False)
    has("cache.subtotal_in_text", csurf, "678.00", want=False)
    has("cache.vat_in_text", csurf, "47.46", want=False)
    # ...while the decoy is everywhere, which is why the failure mode is a
    # confident wrong answer rather than an abstention.
    check("cache.decoy_in_text_count", csurf.count("850.00"), 14)

    # "four minutes later" is derived from the two asset identifiers, not read
    # off the panels: (a) prints its own snapshot id, (b) does not print its
    # own, so the caption attributes the time to the identifier.
    stamps = {}
    for tag, aid in (("cache", CACHE_ASSET), ("scan", SCAN_ASSET)):
        m = re.search(r"_(\d{8})_(\d{6})$", aid)
        check(f"cache.{tag}_id_stamped", bool(m), True)
        stamps[tag] = datetime.datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    check("cache.gap_minutes",
          int((stamps["scan"] - stamps["cache"]).total_seconds() // 60), 4)
    has("cache.cache_prints_own_id", cache_txt, "20241012_084210")
    has("cache.scan_prints_own_id", scan_txt, "20241012_084610", want=False)

    # -- Figure B: the gallery ----------------------------------------------
    for pkg, aid, kind, style in GALLERY:
        hit = [a for a in _manifest(pkg) if a["id"] == aid]
        check(f"gallery.{aid}.found", len(hit), 1)
        if len(hit) != 1:
            continue
        check(f"gallery.{aid}.kind", hit[0].get("kind"), kind)
        check(f"gallery.{aid}.style", _style_of(hit[0]), style)
    check("gallery.n_styles", len({s for _, _, _, s in GALLERY if s}), 5)
    check("gallery.n_packages", len({p for p, _, _, _ in GALLERY}), 6)

    # -- sec:app-assets counts ----------------------------------------------
    defined, nonlegacy = _archetype_library()
    check("arch.defined", len(defined), 18)
    check("arch.sampleable", len(nonlegacy), 16)

    rows = _corpus_assets()
    check("corpus.assets", len(rows), 1568)
    realized = {s for _, s, _ in rows if s}
    check("corpus.styles_realized", len(realized), 14)
    check("corpus.realized_are_sampleable", sorted(realized - set(nonlegacy)), [])
    check("corpus.unsampled", sorted(set(nonlegacy) - realized), sorted(UNSAMPLED))
    kinds = collections.Counter(k for k, _, _ in rows)
    check("corpus.kinds", len(kinds), 5)
    for kind, n in (("chart", 1315), ("rich_text", 205), ("diagram", 28),
                    ("natural_image", 19), ("document", 1)):
        check(f"corpus.kind.{kind}", kinds[kind], n)
    presets = {p for _, _, p in rows if p}
    check("corpus.presets", len(presets), 8)

    return seen, bad


# --- module contract --------------------------------------------------------

def round_example_facts():
    """Summary facts for results_facts.py's SECTIONS/SHIPPED contract.

    Same shape as figure_facts(): the per-claim detail (a hundred-odd entries)
    stays out, what ships is the tally plus the mismatch list, which must be
    empty. A wrong claim lands in the list, a deleted one drops the tally and a
    new unwired one raises it, so none of the three can move silently.
    """
    seen, bad = _compare()
    return {"round_example_checked": len(seen),
            "round_example_mismatches": tuple(bad),
            "round_visible_assets": seen["gate.visible_assets"],
            "round_n_checks": seen["gate.n_checks"],
            "round_gold_steps": seen["gate.gold_steps"],
            "cache_round_visible_assets": seen["cache.visible_assets"],
            "cache_round_n_checks": seen["cache.n_checks"],
            "cache_round_gold_steps": seen["cache.gold_steps"],
            "cache_text_surface_chars": seen["cache.text_surface_chars"],
            "cache_decoy_in_text_count": seen["cache.decoy_in_text_count"],
            "corpus_asset_count": seen["corpus.assets"],
            "corpus_styles_defined": seen["arch.defined"],
            "corpus_styles_sampleable": seen["arch.sampleable"],
            "corpus_styles_realized": seen["corpus.styles_realized"]}


def main():
    seen, bad = _compare()
    for k in sorted(seen):
        print(f"  {k:38} {seen[k]}")
    if bad:
        raise SystemExit("\nROUND EXAMPLE DRIFT\n  " + "\n  ".join(bad))
    print(f"\n  OK: all {len(seen)} claims made by the round-example exhibits "
          f"reproduce from the corpus.")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
