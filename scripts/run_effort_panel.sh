#!/bin/bash

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

#
# run_effort_panel.sh -- submit a reasoning-effort panel as temporary tmux
# sessions, under a machine-wide concurrency cap.
#
# Matrix: 1 corpus x N eval models x 3 aligned effort levels (high/medium/low).
# Defaults describe the v8 corpus (Gemini 3.6-flash generated) x 10 models = 30
# cells; every corpus-specific input is an env var, so a second corpus is a
# different invocation of this same script, NOT a copy of it:
#
#   PANEL_TAG=sonnet5 \
#   TASK_DATA=~/Data/multiagent_conflict_data_Gemini_Sonnet_5 \
#   EVAL_ROOT=~/clash-harness-data/eval_results/data_sonnet_5 \
#   MAX_STEPS=30 RUN_TAG=r02 MAX_PAR=6 ONE_PER_MODEL=1 \
#   SKIP_MODELS=5_5,5_6_sol,5_6_luna,5_6_terra \
#   scripts/run_effort_panel.sh --level medium,low
#
# PANEL_TAG is mandatory for a second corpus: the cell key is (level, slug)
# only, so without it two panels share log files, launcher scripts and tmux
# session names. See the PANEL_TAG block below.
#
# WHY `low` AND NOT `minimal` FOR THE BOTTOM RUNG
# -----------------------------------------------
# `--llm-reasoning-effort` folds one shared ladder onto each vendor's wire
# vocabulary (train_eval/agents.py:972-1024). At the bottom the vendors DIVERGE:
#
#     flag       Gemini thinking_level   OpenAI effort   Anthropic effort
#     minimal    MINIMAL                 low             low
#     low        LOW                     low             low
#
# So `minimal` is the one rung that is NOT an equal-effort comparison: the Flash
# models would run at MINIMAL while every Claude/GPT model runs at `low`. Worse,
# gemini-3.1-pro rejects MINIMAL with a 400 and is silently clamped UP to LOW
# (train_eval/_constants.py:106-112), so a `minimal` panel runs at three
# different effective levels at once. `low` maps to low/LOW everywhere, which is
# what "aligned_setting" is supposed to mean.
#
# SCHEDULING
# ----------
# Three gates, all re-read on every poll, all machine-wide:
#   1. process count -- counts EVERY train_and_eval on the host, not just the
#      ones this script started, so it waits behind hand-launched tmux sessions
#      instead of piling on top of them.
#   2. MemAvailable floor -- a count cap alone is not enough, because a run's
#      RSS grows with its age (measured here: 7.6G at 3h vs 1.5-2.5G at 15min).
#      N slots that fit at launch can still exhaust the host later.
#   3. swap I/O rate -- and NOT "is swap full". Swap on this host sits pinned at
#      100% for days because Linux never reclaims a swapped page until it is
#      touched, so fullness is a high-water mark, not pressure.
#
# SUPERVISION
# -----------
# The queue is rebuilt from the filesystem + process table on every pass, and
# the script keeps looping until every requested cell has a summary. A run that
# dies without writing one therefore gets relaunched, up to MAX_ATTEMPTS, after
# which that cell is written off so one poison cell cannot spin the loop.
#
# IDEMPOTENT
# ----------
# A cell is skipped when it has a finished `parallel_run.summary`, or when a
# live process already owns its --out-dir. Re-running this script therefore
# tops up whatever is missing; it never double-submits. Partially-finished
# cells have no summary and are relaunched from scratch (--clean-out-dir).
#
# USAGE
#   scripts/run_effort_panel.sh --status              # print the 30-cell matrix
#   scripts/run_effort_panel.sh --dry-run             # print pending commands
#   scripts/run_effort_panel.sh                       # supervise (blocks)
#   scripts/run_effort_panel.sh --detach              # supervise in background
#   scripts/run_effort_panel.sh --level medium,low    # subset of levels
#   MAX_PAR=8 MIN_AVAIL_GB=25 scripts/run_effort_panel.sh   # tighten the gates
#
set -u -o pipefail

# ---------------------------------------------------------------- config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

VENV="${VENV:-$HOME/Code/multi-agent}"
GEMINI_ENV="${GEMINI_ENV:-$HOME/.config/gemini_env.sh}"
TASK_DATA="${TASK_DATA:-$HOME/Data/multiagent_conflict_data_v8_Gemini_3_6_flash}"
EVAL_ROOT="${EVAL_ROOT:-$HOME/clash-harness-data/eval_results/data_3_6_flash_v8}"
LOG_DIR="${LOG_DIR:-$HOME/eval_logs}"
LAUNCH_DIR="$LOG_DIR/launchers"
SCHED_LOG="$LOG_DIR/effort_panel_scheduler.log"

MAX_PAR="${MAX_PAR:-10}"      # machine-wide cap on concurrent train_and_eval
POLL_S="${POLL_S:-60}"        # seconds between capacity checks
SETTLE_S="${SETTLE_S:-20}"    # pause after a launch so the child registers in
                              # the process table before the next capacity read
RUN_TAG="${RUN_TAG:-r01}"     # seed/repeat directory

# Step budget for every cell. A panel is only internally comparable if all of
# its cells share one budget, so this is deliberately ONE knob rather than a
# per-model column. The v8 panel ran at 20; the Sonnet-5 corpus needs 30,
# because at 20 five of its six arms hit the ceiling on 38-93% of rounds and a
# truncated arm measures the budget, not the model.
MAX_STEPS="${MAX_STEPS:-20}"

# Modality arm. Empty means "pass nothing", so the harness keeps its own
# default of `both` and an unset variable reproduces the shipped panels
# byte-for-byte -- verified with --dry-run against the v8 launchers before this
# was added. Set to `path` for a blind arm: assets stay reachable through the
# tools but no pixels enter the request.
#
# `path` is a real blind condition, not a downgrade. The asset backend's
# allowlist (clasharness/backend/asset_backend.py) withholds description,
# caption, ocr_text and chart_rows, `asset read-text` returns an explicit "no
# extractable text" stub, and the v8 corpus's specs.json carries none of those
# fields anyway.
#
# TWO THINGS A BLIND ARM DOES NOT CONTROL FOR, both deliberate:
#   1. The task instruction still tells the agent to run `asset list` and
#      `asset open`. It is baked into the corpus. Leaving it in is what keeps
#      this a ONE-variable manipulation -- strip it and the arm varies the
#      prompt as well as the pixels.
#   2. 346 of the 3,760 grading checks are assets_accessed/required, and they
#      are satisfied by CALLING `asset open`, which still succeeds blind. The
#      blind arm can therefore still bank those points; score it both raw and
#      with those checks excluded.
MULTIMODAL_MODE="${MULTIMODAL_MODE:-}"

# Namespaces the per-cell log file, launcher script and tmux session name.
# WITHOUT this, two panels over different corpora collide: the cell key is
# (level, slug) only, so a Sonnet-5 cell would append to the v8 cell's log,
# `cell_last_exit` would read the OTHER panel's EXIT= line and give up on a
# healthy cell, and `submit` would see the other panel's tmux session and skip
# forever. (`cell_running` is already safe -- it keys on --out-dir, which does
# carry the corpus.)
PANEL_TAG="${PANEL_TAG:-v8}"

# Serialize cells that share a model: at most one live cell per model id.
# `model_busy` below normally only does this for models throttled BELOW
# DEFAULT_WORKERS, on the theory that a model at full workers has quota to
# spare. That theory does not survive a provider running hot: gemini-3.1-pro
# returned 504 DEADLINE_EXCEEDED with qos=SHEDDABLE_PLUS on 18/570 tasks of a
# single 40-worker run -- server-side load shedding, which no per-model quota
# column predicts. Set to 1 to spread one model's cells out in time instead.
ONE_PER_MODEL="${ONE_PER_MODEL:-0}"

# Resource gates. A process count alone is NOT enough: a run's RSS grows with
# its age (measured on this host: a 3h-old run held 7.6G while 15min-old runs
# held 1.5-2.5G), so N slots that are comfortable at launch can still exhaust
# the host three hours later. Both gates below are re-evaluated on every poll,
# so the scheduler simply stops feeding the machine when it tightens up.
MIN_AVAIL_GB="${MIN_AVAIL_GB:-20}"   # need this much MemAvailable to launch
MAX_SWAP_KBPS="${MAX_SWAP_KBPS:-20480}"  # sustained swap I/O that counts as
                                          # thrashing (KB/s in+out; <=0 disables)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"    # relaunches per cell before giving up

# A run whose tasks mostly died on provider errors is not a result, it is an
# outage. Above this percentage the cell is treated as unfinished and relaunched
# rather than accepted. 5% comfortably clears the incidental noise seen on clean
# runs (high/opus_5 landed 3/600 = 0%) while catching a wipeout.
MAX_PROVFAIL_PCT="${MAX_PROVFAIL_PCT:-5}"

DEFAULT_WORKERS="${DEFAULT_WORKERS:-20}"

# Comma-separated slugs to leave out of the panel entirely -- for a model that
# is blocked on something this scheduler cannot fix, so it neither burns a
# concurrency slot nor accumulates doomed retry attempts. The MODELS row stays
# put so the cell reappears the moment the blocker clears.
#
# Currently expected: 5_5. openai.gpt-5.5's bedrock input-tpm quota on project
# 289334605194 rejects v8 requests outright -- every 429 reports the SAME
# `exceeded by 4285`, unchanged between 20 workers (994 samples) and 3 workers
# (12 samples), so the overage is per-request, not rate-driven, and no amount of
# throttling clears it. The client already backs off 15/30/60s x20 retries
# (llm_client.py:462) and still lands 0 clean tasks out of 9. Note this model ran
# clean on the base and v3 corpora (provider_failures=0, 0, 0 and 1), so it needs
# a quota increase rather than a code change.
SKIP_MODELS="${SKIP_MODELS:-}"

JUDGE_MODEL="gemini-3.6-flash"
JUDGE_PROVIDER="gemini"

# Ordered vendor round-robin so we never hammer one provider's quota with 4
# simultaneous cold starts.
#
# The trailing `workers` column is --parallel-workers for that model. It is a
# harness throughput knob only -- examples/train_and_eval.py:1520 feeds it
# straight to ThreadPoolExecutor(max_workers=...) and nothing else reads it, so
# lowering it changes wall-clock but not a single prompt, model setting or judge
# call. Measured capability is unaffected and cross-model comparison stays fair.
#
# These rows are the ten API models the paper reports, and they must stay in
# step with `REPORT_DIRS` in analysis/corpus.py -- that tuple is what the tables
# are computed over, so a model here that is missing there is a run nobody reads,
# and one there that is missing here cannot be reproduced by this script.
#
# The published runs went through Vertex AI (Claude, Gemini) and Bedrock (GPT).
# The rows below instead name each vendor's own API, because that is what a
# reader can obtain: `anthropic` (ANTHROPIC_API_KEY), `gemini` (GEMINI_API_KEY)
# and `openai` (OPENAI_API_KEY). The model ids are identical on both routes
# except that Bedrock prefixes GPT ids with `openai.`, which is dropped here.
# To reproduce the original routing instead, set the provider column back to
# `vertex-anthropic` / `bedrock-gpt` and fill in the vertex_location column;
# both providers are still supported.
#
# gpt-5.5 is deliberately absent. Its run aborted with 100% provider_failure
# rows (its Bedrock input-tpm quota is per-model and far below its siblings':
# at 20 workers it 429'd from the 47th log line on and lost 600/600 tasks at
# high), so it contributes no data and analysis/corpus.py lists it in
# EXCLUDED_MODELS. Running it here would only burn quota producing rows the
# analysis discards.
#
#   slug | Vendor | provider | model | vertex_location | workers
MODELS=(
  "3_6_flash|Gemini|gemini|gemini-3.6-flash||20"
  "opus_4_8|Claude|anthropic|claude-opus-4-8||20"
  "3_7_flash|Gemini|gemini|gemini-3.7-flash||20"
  "3_5_flash|Gemini|gemini|gemini-3.5-flash||20"
  "opus_5|Claude|anthropic|claude-opus-5||20"
  "5_6_sol|OpenAI|openai|gpt-5.6-sol||20"
  "3_1_pro|Gemini|gemini|gemini-3.1-pro-preview||20"
  "sonnet_5|Claude|anthropic|claude-sonnet-5||20"
  "5_6_luna|OpenAI|openai|gpt-5.6-luna||20"
  "5_6_terra|OpenAI|openai|gpt-5.6-terra||20"
)

ALL_LEVELS=(high medium low)

# ------------------------------------------------------------------ args ----
LEVELS=()
DRY_RUN=0
STATUS_ONLY=0
DETACH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=1 ;;
    --status)   STATUS_ONLY=1 ;;
    --detach)   DETACH=1 ;;
    --level)    shift; IFS=',' read -r -a _lv <<<"${1:-}"; LEVELS+=("${_lv[@]}") ;;
    --level=*)  IFS=',' read -r -a _lv <<<"${1#*=}"; LEVELS+=("${_lv[@]}") ;;
    --max-par)  shift; MAX_PAR="${1:-10}" ;;
    --max-par=*) MAX_PAR="${1#*=}" ;;
    # Prints the header block: everything between the shebang and `set -u`,
    # located at runtime so editing the header cannot silently truncate --help.
    -h|--help)  sed -n "2,$(($(grep -n '^set -u' "${BASH_SOURCE[0]}" | head -1 | cut -d: -f1) - 1))p" "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ ${#LEVELS[@]} -eq 0 ]] && LEVELS=("${ALL_LEVELS[@]}")

for lv in "${LEVELS[@]}"; do
  case "$lv" in
    high|medium|low) ;;
    *) echo "bad --level '$lv' (want high|medium|low)" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR" "$LAUNCH_DIR"

# --------------------------------------------------------------- helpers ----
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

out_dir_for() {  # level slug vendor
  echo "$EVAL_ROOT/aligned_setting_$1/$3/$2/$RUN_TAG"
}

# A cell is finished when train_and_eval wrote its end-of-run summary. The file
# only appears on a completed run, so a killed/partial run correctly reads as
# not-done and gets relaunched under --clean-out-dir.
# A summary file is necessary but NOT sufficient. train_and_eval writes one even
# when every single task died on provider errors: the v8 high/5_5 cell landed
# `Tasks : 0/600 passed  raw_accuracy=0.0%` with `provider_failures=600`, which
# an existence check happily reports as finished -- a fabricated 0.0% that would
# have gone straight into the panel. So a cell only counts as done when its
# provider-failure rate is under the tolerance too, which puts the relaunch on
# the scheduler instead of on someone noticing afterwards.
cell_done() {  # out_dir
  local s="$1/parallel_run.summary"
  [[ -s "$s" ]] || return 1
  grep -q '^Tasks :' "$s" 2>/dev/null || return 1

  local tot pf
  tot="$(grep -oE '^Tasks : [0-9]+/[0-9]+' "$s" | head -1 | sed 's|.*/||')"
  pf="$(grep -oE 'provider_failures=[0-9]+' "$s" | head -1 | cut -d= -f2)"

  # Missing/unparseable fields: fall back to the old behaviour rather than
  # relaunching a run we cannot judge.
  [[ "$tot" =~ ^[0-9]+$ && "$tot" -gt 0 ]] || return 0
  [[ "$pf"  =~ ^[0-9]+$ ]] || return 0

  (( pf * 100 > tot * MAX_PROVFAIL_PCT )) && return 1
  return 0
}

# Human-readable verdict for --status, so a tainted cell is visible rather than
# just silently re-queued.
cell_provfail_pct() {  # out_dir -> integer percent, or "" when unknown
  local s="$1/parallel_run.summary" tot pf
  [[ -s "$s" ]] || return
  tot="$(grep -oE '^Tasks : [0-9]+/[0-9]+' "$s" | head -1 | sed 's|.*/||')"
  pf="$(grep -oE 'provider_failures=[0-9]+' "$s" | head -1 | cut -d= -f2)"
  [[ "$tot" =~ ^[0-9]+$ && "$tot" -gt 0 && "$pf" =~ ^[0-9]+$ ]] || return
  echo $(( pf * 100 / tot ))
}

# Exit status of this cell's most recent launch, from the trailing EXIT= line
# the launcher appends. Echoes "?" when the log has none (never launched, or
# killed hard enough that the launcher never got to write it).
cell_log_for() {  # level slug
  echo "$LOG_DIR/${PANEL_TAG}_${1}_${2}.log"
}

cell_last_exit() {  # level slug
  local log; log="$(cell_log_for "$1" "$2")"
  [[ -r "$log" ]] || { echo '?'; return; }
  local e
  e="$(grep -o '^EXIT=[0-9]\+' "$log" 2>/dev/null | tail -1)"
  echo "${e#EXIT=}" | grep -q '[0-9]' && echo "${e#EXIT=}" || echo '?'
}

# Single snapshot of the process table, refreshed explicitly. Taking it up front
# and matching against the snapshot IN-SHELL is not an optimisation: piping ps
# into `grep -F -- "--out-dir $out"` makes grep's own argv contain the out-dir,
# ps lists that argv, and every cell reports itself as already running.
PS_SNAPSHOT=""
refresh_ps() { PS_SNAPSHOT="$(ps -eo args= 2>/dev/null)"; }

# Someone (this script, or a hand-launched tmux session) is already driving
# this out-dir. Demand a delimiter after the path so .../r01 does not match
# a hypothetical .../r010.
cell_running() {  # out_dir
  [[ "$PS_SNAPSHOT" == *"--out-dir $1 "* ]] && return 0
  [[ "$PS_SNAPSHOT" == *"--out-dir $1"$'\n'* ]] && return 0
  [[ "$PS_SNAPSHOT" == *"--out-dir $1" ]]
}

# Machine-wide count of live eval processes, from the same snapshot.
running_count() {
  printf '%s\n' "$PS_SNAPSHOT" | grep -cE 'python3? .*train_and_eval[.]py' || true
}

# MemAvailable, not free: it already nets out the reclaimable page cache, which
# on this host is ~40G and would otherwise make the machine look full when it
# is not.
mem_avail_gb() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }

# The token quotas that wiped out gpt-5.5 are keyed per MODEL (the 429 names
# `input-tpm:<project>:openai.gpt-5.5`), so that model's high/medium/low cells
# all draw on ONE bucket. Running two of its cells at once would double the
# request rate and put it straight back over the limit that we just throttled it
# under -- so a model we had to throttle gets one live cell at a time. Models at
# full workers are unconstrained and keep overlapping freely.
model_busy() {  # model_id workers
  [[ "$ONE_PER_MODEL" == 1 ]] || { [[ "$2" -ge "$DEFAULT_WORKERS" ]] && return 1; }
  [[ "$PS_SNAPSHOT" == *"--model $1 "* ]]
}

model_skipped() {  # slug
  [[ -n "$SKIP_MODELS" ]] || return 1
  [[ ",$SKIP_MODELS," == *",$1,"* ]]
}

# Sustained swap traffic over a short window, in KB/s. Deliberately NOT keyed on
# "SwapFree == 0": swap on this host has been pinned at 100% for days by
# long-lived desktop processes' cold pages, and Linux never reclaims a swapped
# page until it is touched again, so fullness says nothing about pressure. Rate
# does. Caveat: once swap is full the kernel cannot page out any more, so
# pswpout flatlines and pswpin carries the signal -- which is why MemAvailable
# above, not this, is the primary gate.
swap_kbps() {  # window_s
  local w="${1:-3}" a b
  a=$(awk '/^pswpin |^pswpout /{s+=$2} END{print s+0}' /proc/vmstat)
  sleep "$w"
  b=$(awk '/^pswpin |^pswpout /{s+=$2} END{print s+0}' /proc/vmstat)
  awk -v a="$a" -v b="$b" -v w="$w" 'BEGIN{printf "%d", (b-a)*4/w}'
}

# One combined verdict. Echoes a human-readable reason on stdout when blocked,
# nothing when clear; return code carries the decision.
capacity_block_reason() {  # live_count
  local live="$1" avail sw
  [[ "$live" -ge "$MAX_PAR" ]] && { echo "host at $live/$MAX_PAR live evals"; return 0; }
  avail="$(mem_avail_gb)"
  [[ "$avail" -lt "$MIN_AVAIL_GB" ]] && { echo "MemAvailable ${avail}G < ${MIN_AVAIL_GB}G floor (live=$live)"; return 0; }
  if [[ "$MAX_SWAP_KBPS" -gt 0 ]]; then
    sw="$(swap_kbps 3)"
    [[ "$sw" -gt "$MAX_SWAP_KBPS" ]] && { echo "swap I/O ${sw}KB/s > ${MAX_SWAP_KBPS}KB/s (live=$live, avail=${avail}G)"; return 0; }
  fi
  return 1
}

build_cmd() {  # level slug vendor provider model vloc workers -> sets CMD array
  local level="$1" slug="$2" vendor="$3" provider="$4" model="$5" vloc="$6"
  local workers="${7:-$DEFAULT_WORKERS}"
  local out; out="$(out_dir_for "$level" "$slug" "$vendor")"

  CMD=(python3 -u examples/train_and_eval.py
       --agent llm
       --llm-provider "$provider"
       --model "$model")
  [[ -n "$vloc" ]] && CMD+=(--vertex-location "$vloc")

  CMD+=(--llm-judge-model "$JUDGE_MODEL"
        --llm-judge-provider "$JUDGE_PROVIDER"
        --llm-max-tokens 32768
        --split total
        --task-data-dir "$TASK_DATA"
        --collect-data
        --out-dir "$out"
        --llm-timeout-s 180
        --parallel-workers "$workers"
        --llm-request-retries 20
        --max-steps "$MAX_STEPS"
        --clean-out-dir
        --verbose
        --capture-log
        --no-compact-fallback
        --llm-tool-mode native)

  # Unset by default, so this appends nothing and the shipped panels are
  # reproduced exactly. See the MULTIMODAL_MODE note in the config block.
  [[ -n "$MULTIMODAL_MODE" ]] && CMD+=(--multimodal-input-mode "$MULTIMODAL_MODE")

  # Per-vendor image settings. All three target a 2576px long edge; for Gemini
  # the edge flag alone is a no-op on what the model actually resolves, which is
  # why the media-resolution flag has to ride along with it.
  case "$vendor" in
    Gemini) CMD+=(--gemini-max-image-edge 2576 --gemini-media-resolution ultra_high) ;;
    Claude) CMD+=(--anthropic-max-image-edge 2576) ;;
    OpenAI) CMD+=(--openai-max-image-edge 2576) ;;
  esac

  # NOTE: --retry-provider-failures-only is deliberately NOT passed. It is
  # mutually exclusive with --clean-out-dir, and this script's idempotency comes
  # from the cell_done/cell_running checks instead.
  CMD+=(--llm-reasoning-effort "$level")
}

submit() {  # level slug vendor provider model vloc workers
  local level="$1" slug="$2" vendor="$3"
  local out log launcher sess
  out="$(out_dir_for "$level" "$slug" "$vendor")"
  log="$(cell_log_for "$level" "$slug")"
  launcher="$LAUNCH_DIR/${PANEL_TAG}_${level}_${slug}.sh"
  sess="ev_${PANEL_TAG}_${level}_${slug}"

  build_cmd "$@"

  if tmux has-session -t "$sess" 2>/dev/null; then
    say "SKIP  $level/$slug -- tmux session '$sess' already exists"
    return 1
  fi

  mkdir -p "$(dirname "$out")"

  {
    echo '#!/bin/bash'
    echo '# generated by run_effort_panel.sh -- safe to delete when idle'
    echo 'set -u -o pipefail'
    printf 'source %q/bin/activate\n' "$VENV"
    printf 'source %q\n' "$GEMINI_ENV"
    printf '{ echo; echo "=== START $(date -Is) %s/%s ==="; } >> %q\n' "$level" "$slug" "$log"
    # A repo that cannot be entered (unmounted share, revoked credentials on a
    # network filesystem) is an infrastructure outage, not a broken cell. Say so
    # in the log instead of dying silently -- the cell stays pending and the next
    # pass of this script picks it up once the path is readable again.
    printf 'if ! cd %q; then echo "EXIT=97 (cannot cd to repo)" >> %q; exit 97; fi\n' "$REPO" "$log"
    echo 'export PYTHONPATH=.'
    printf '%q ' "${CMD[@]}"; printf '>> %q 2>&1\n' "$log"
    printf 'rc=$?\n'
    printf 'echo "EXIT=$rc" >> %q\n' "$log"
    printf 'exit $rc\n'
  } > "$launcher"
  chmod +x "$launcher"

  # The session is created with the launcher as its command, so tmux tears the
  # session down by itself the moment the run exits -- no reaper needed.
  if tmux new-session -d -s "$sess" "bash $launcher"; then
    say "START $level/$slug  -> $out"
    return 0
  fi
  say "FAIL  $level/$slug -- tmux new-session failed"
  return 1
}

print_status() {
  local total=0 done_n=0 run_n=0 pend_n=0
  refresh_ps
  printf '\n%-10s %-12s %-9s %s\n' LEVEL MODEL STATE OUT
  printf '%s\n' "----------------------------------------------------------------------"
  for level in "${ALL_LEVELS[@]}"; do
    for row in "${MODELS[@]}"; do
      IFS='|' read -r slug vendor _p _m _v _w <<<"$row"
      local out state
      out="$(out_dir_for "$level" "$slug" "$vendor")"
      total=$((total+1))
      if model_skipped "$slug"; then state=SKIPPED
      elif cell_done "$out"; then state=DONE; done_n=$((done_n+1))
      elif cell_running "$out"; then state=RUNNING; run_n=$((run_n+1))
      else
        # A cell that has a summary but still is not done failed the provider
        # -failure tolerance: say so, so it is not mistaken for never-launched.
        local pct; pct="$(cell_provfail_pct "$out")"
        if [[ -n "$pct" ]]; then state="REDO:${pct}%pf"; else state=pending; fi
        pend_n=$((pend_n+1))
      fi
      printf '%-10s %-12s %-9s %s\n' "$level" "$slug" "$state" "${out/#$HOME/\~}"
    done
  done
  printf '%s\n' "----------------------------------------------------------------------"
  printf 'total=%d  DONE=%d  RUNNING=%d  pending=%d\n' \
    "$total" "$done_n" "$run_n" "$pend_n"
  printf 'host: %s/%s live evals | MemAvailable %sG (floor %sG) | swap %sKB/s (max %s)\n\n' \
    "$(running_count)" "$MAX_PAR" "$(mem_avail_gb)" "$MIN_AVAIL_GB" \
    "$(swap_kbps 2)" "$MAX_SWAP_KBPS"
}

# ------------------------------------------------------------------ main ----
if [[ $STATUS_ONLY -eq 1 ]]; then
  print_status
  exit 0
fi

if [[ $DETACH -eq 1 && "${_EFFORT_PANEL_DETACHED:-0}" != "1" ]]; then
  args=(--max-par "$MAX_PAR")
  for lv in "${LEVELS[@]}"; do args+=(--level "$lv"); done
  _EFFORT_PANEL_DETACHED=1 setsid nohup "${BASH_SOURCE[0]}" "${args[@]}" \
    >> "$SCHED_LOG" 2>&1 &
  echo "scheduler detached: pid=$! log=$SCHED_LOG"
  exit 0
fi

declare -A ATTEMPTS=()   # cell -> launches this scheduler has made
declare -A GAVEUP=()     # cell -> reason, once it burned MAX_ATTEMPTS

# Pending queue in submission order: level-major (high first, so the headline
# table completes first), vendor round-robin within a level. Rebuilt every pass
# from the live filesystem + process table, so a run that died without writing a
# summary reappears here and gets relaunched. Refreshes PS_SNAPSHOT as a side
# effect; callers rely on that.
build_queue() {
  QUEUE=()
  refresh_ps
  local level row slug vendor provider model vloc out key
  for level in "${LEVELS[@]}"; do
    for row in "${MODELS[@]}"; do
      IFS='|' read -r slug vendor provider model vloc workers <<<"$row"
      out="$(out_dir_for "$level" "$slug" "$vendor")"
      key="$level/$slug"
      model_skipped "$slug" && continue
      cell_done "$out" && continue
      cell_running "$out" && continue
      [[ -n "${GAVEUP[$key]:-}" ]] && continue
      QUEUE+=("$level|$slug|$vendor|$provider|$model|$vloc|${workers:-$DEFAULT_WORKERS}")
    done
  done
}

# Requested cells that are neither finished nor written off. Zero means the
# scheduler's job is over; anything else means keep watching.
outstanding() {
  local n=0 level row slug vendor out key
  for level in "${LEVELS[@]}"; do
    for row in "${MODELS[@]}"; do
      IFS='|' read -r slug vendor _ _ _ _ <<<"$row"
      out="$(out_dir_for "$level" "$slug" "$vendor")"
      key="$level/$slug"
      model_skipped "$slug" && continue
      cell_done "$out" && continue
      [[ -n "${GAVEUP[$key]:-}" ]] && continue
      n=$((n+1))
    done
  done
  echo "$n"
}

build_queue

if [[ $DRY_RUN -eq 1 ]]; then
  say "queued ${#QUEUE[@]} run(s); levels=${LEVELS[*]}; cap=$MAX_PAR"
  i=0
  for item in "${QUEUE[@]}"; do
    IFS='|' read -r level slug vendor provider model vloc workers <<<"$item"
    build_cmd "$level" "$slug" "$vendor" "$provider" "$model" "$vloc" "$workers"
    i=$((i+1))
    printf '\n--- [%02d] %s / %s (%s) ---\n' "$i" "$level" "$slug" "$vendor"
    printf '%q ' "${CMD[@]}"; printf '\n'
  done
  printf '\n'
  say "dry-run only; nothing submitted"
  exit 0
fi

if [[ "$(outstanding)" -eq 0 ]]; then
  say "nothing to do -- every requested cell is already DONE"
  print_status
  exit 0
fi

say "supervising levels=${LEVELS[*]}  cap=$MAX_PAR evals  floor=${MIN_AVAIL_GB}G avail" \
    " swap<=${MAX_SWAP_KBPS}KB/s  poll=${POLL_S}s  max_attempts=$MAX_ATTEMPTS"
say "outstanding=$(outstanding) cells; ${#QUEUE[@]} launchable right now"

PASS=0
while :; do
  PASS=$((PASS+1))

  for item in "${QUEUE[@]}"; do
    IFS='|' read -r level slug vendor provider model vloc workers <<<"$item"
    key="$level/$slug"

    # Hold here until the host has room. Both gates are re-read every poll, so a
    # machine that tightens up mid-queue stops receiving work rather than the
    # scheduler charging ahead on a stale reading.
    while :; do
      refresh_ps
      reason="$(capacity_block_reason "$(running_count)")" || break
      say "wait  $key -- $reason"
      sleep "$POLL_S"
    done

    # Re-check right before launching: a concurrent operator (or the previous
    # pass) may have taken this cell while we were queued.
    out="$(out_dir_for "$level" "$slug" "$vendor")"
    if cell_done "$out"; then say "SKIP  $key -- finished while queued"; continue; fi
    if cell_running "$out"; then say "SKIP  $key -- started elsewhere while queued"; continue; fi

    # Deferred, not waited on: a throttled model can be blocked for hours by its
    # own sibling cell, and blocking here would stall every unrelated cell behind
    # it in the queue. The next pass picks it up.
    if model_busy "$model" "$workers"; then
      say "DEFER $key -- sibling $slug cell live, shared per-model token quota"
      continue
    fi

    if submit "$level" "$slug" "$vendor" "$provider" "$model" "$vloc" "$workers"; then
      ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
      say "      $key attempt ${ATTEMPTS[$key]}/$MAX_ATTEMPTS"
      # Let the child register in the process table before the next capacity
      # read, otherwise the whole queue slips through one gate check.
      sleep "$SETTLE_S"
    else
      say "DEFER $key -- will retry next pass"
    fi
  done

  # A cell that has burned its attempts and is still not running is written off,
  # so one poison cell cannot spin the supervisor forever.
  refresh_ps
  for level in "${LEVELS[@]}"; do
    for row in "${MODELS[@]}"; do
      IFS='|' read -r slug vendor _ _ _ _ <<<"$row"
      key="$level/$slug"
      out="$(out_dir_for "$level" "$slug" "$vendor")"
      model_skipped "$slug" && continue
      cell_done "$out" && continue
      cell_running "$out" && continue
      [[ -n "${GAVEUP[$key]:-}" ]] && continue
      [[ "${ATTEMPTS[$key]:-0}" -ge "$MAX_ATTEMPTS" ]] || continue

      # EXIT=97 is the launcher failing to cd into the repo. That is an
      # infrastructure outage, not a broken cell, and it will hit every cell that
      # happens to launch during the gap -- burning all three attempts in three
      # minutes and writing off the whole remaining panel. Refund it and keep
      # retrying; the run becomes launchable again the moment the path is
      # readable, with no operator intervention beyond that.
      if [[ "$(cell_last_exit "$level" "$slug")" == "97" ]]; then
        ATTEMPTS[$key]=$(( MAX_ATTEMPTS - 1 ))
        say "HOLD  $key -- EXIT=97, repo path unreadable. Not counted."
        continue
      fi

      GAVEUP[$key]="failed ${ATTEMPTS[$key]}x, last EXIT=$(cell_last_exit "$level" "$slug")"
      say "GIVEUP $key -- ${GAVEUP[$key]}; see $(cell_log_for "$level" "$slug")"
    done
  done

  left="$(outstanding)"
  if [[ "$left" -eq 0 ]]; then
    say "ALL DONE after $PASS pass(es)"
    break
  fi

  sleep "$POLL_S"
  build_queue
  if [[ ${#QUEUE[@]} -gt 0 ]]; then
    say "pass $PASS: $left outstanding, ${#QUEUE[@]} relaunchable (a run exited without a summary)"
  fi
done

if [[ ${#GAVEUP[@]} -gt 0 ]]; then
  say "WROTE OFF ${#GAVEUP[@]} cell(s):"
  for k in "${!GAVEUP[@]}"; do say "  $k -- ${GAVEUP[$k]}"; done
fi
print_status
