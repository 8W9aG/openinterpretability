#!/usr/bin/env bash
# Run experiments back to back, each in a fresh headless Claude Code session, until
# the iteration limit is reached or `claude` fails (e.g. the usage limit is hit).
#
# Claude drives the workflow (land-experiment: plan, claim, audit, review, post a PR,
# wait for green CI, and merge it when the PR only touches wiki/ and `gh` has write
# access); the proposer and reviewer LLMs are whatever you pass here. A model
# `external/<name>` means the driving Claude session answers that agent itself.
#
# Runs in a separate git worktree on a fresh origin/main each iteration, so your
# own checkout is never touched. Usage: scripts/experiment-loop.sh --help

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/experiment-loop.sh [options]

  --proposer MODEL     LiteLLM model or external/<name> for `open-interpretability audit`
                       (default: $OPENINTERP_AGENT_MODEL, else anthropic/claude-opus-5-5)
  --reviewer MODEL     LiteLLM model or external/<name> for `open-interpretability verify`
                       (default: same as the proposer; records are then not independent)
  --driver MODEL       Claude model that drives each session (default: claude-haiku-5-5;
                       '' for claude's own; see docs/analyses/driver-model-pilot.md)
  --corpus PATH        Scan corpus (default: each new scan builds a fresh `mixed` corpus)
  --max N              Stop after N experiments; 0 = until claude fails (default: 0)
  --epsilon P          Exploration probability passed to `open-interpretability next` (default: 0.7)
  --worktree DIR       Worktree to run in (default: ../<repo>-loop, created if missing)
  --max-turns N        Turn limit per session, a guard against runaway sessions (default: 200)
  --budget USD         Stop once this loop has spent USD or more (default: no cap)
  --daily-budget USD   Stop once all loops together have spent USD or more today, UTC,
                       going by the logs in <repo>/logs/loop/ (default: no cap)
  --one-heavy-job      Queue model loads and `make test` behind a machine-wide lock, so
                       loops in other checkouts don't run them at the same time (opt-in;
                       openinterp/core/heavy_lock.py)
  -h, --help           Show this help

Budgets count the dollars that `open-interpretability costs --usd` reports: priced LLM calls and
each session's own cost. The loop checks them between experiments, never during one, so
the experiment that crosses a cap still finishes and posts. Calls with no price are not
counted, and the loop warns when there are some.

To stop every loop on this machine after its current experiment, create the stop file:
  touch "${XDG_CACHE_HOME:-$HOME/.cache}/openinterp/stop"   (or $OPENINTERP_STOP_FILE)
Remove it before you start a loop again.

Logs go to <repo>/logs/loop/: per experiment, the session's stderr and result (.log),
its `claude --output-format json` result (.json), and the costs ledger every openinterp
command appends to (.costs.jsonl). After each experiment the loop prints what it cost
and a running total (`open-interpretability costs`). Set CLAUDE_BIN if `claude` is not on PATH.
EOF
}

REPO=$(git rev-parse --show-toplevel)
PROPOSER=${OPENINTERP_AGENT_MODEL:-anthropic/claude-opus-5-5}
REVIEWER=""
DRIVER=claude-haiku-5-5 # lands as reliably as larger drivers at a fraction of the cost
CORPUS=""
MAX=0
EPSILON=0.7
WORKTREE="$(dirname "$REPO")/$(basename "$REPO")-loop"
MAX_TURNS=200
BUDGET=""
DAILY_BUDGET=""
CLAUDE_BIN=${CLAUDE_BIN:-claude}

while [[ $# -gt 0 ]]; do
  case $1 in
    --proposer) PROPOSER=$2; shift 2 ;;
    --reviewer) REVIEWER=$2; shift 2 ;;
    --driver) DRIVER=$2; shift 2 ;;
    --corpus) CORPUS=$2; shift 2 ;;
    --max) MAX=$2; shift 2 ;;
    --epsilon) EPSILON=$2; shift 2 ;;
    --worktree) WORKTREE=$2; shift 2 ;;
    --max-turns) MAX_TURNS=$2; shift 2 ;;
    --budget) BUDGET=$2; shift 2 ;;
    --daily-budget) DAILY_BUDGET=$2; shift 2 ;;
    --one-heavy-job) export OPENINTERP_HEAVY_LOCK=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
REVIEWER=${REVIEWER:-$PROPOSER}
# `next --steps` writes the subagent flow for external/ models, one command for API models.
STEP_MODELS=""
[[ $PROPOSER != external/* ]] && STEP_MODELS+=" --proposer $PROPOSER"
[[ $REVIEWER != external/* ]] && STEP_MODELS+=" --reviewer $REVIEWER"

command -v "$CLAUDE_BIN" >/dev/null || { echo "claude not found; set CLAUDE_BIN" >&2; exit 1; }
command -v gh >/dev/null || { echo "gh is required to claim and post experiments" >&2; exit 1; }
CORPUS_OPTION="--fresh-corpus"
if [[ -n $CORPUS ]]; then
  [[ -f $CORPUS ]] || { echo "corpus not found: $CORPUS" >&2; exit 1; }
  CORPUS_OPTION="--corpus $(cd "$(dirname "$CORPUS")" && pwd)/$(basename "$CORPUS")"
fi
if [[ $REVIEWER == "$PROPOSER" ]]; then
  echo "warning: proposer and reviewer are both $PROPOSER, so every review will be badged" \
    "self-reviewed; pass --reviewer <another model> for independent reviews" >&2
fi

if [[ ! -d $WORKTREE ]]; then
  git -C "$REPO" fetch --quiet origin main
  git -C "$REPO" worktree add --detach "$WORKTREE" origin/main
fi

LOG_DIR="$REPO/logs/loop"
mkdir -p "$LOG_DIR"

# Tools the session may use without asking. Nothing outside the repo workflow.
ALLOWED_TOOLS=(
  Read Edit Write Glob Grep Skill Agent
  "Bash(uv run:*)" "Bash(make:*)" "Bash(git:*)" "Bash(gh pr:*)" "Bash(gh run:*)" "Bash(gh repo view:*)"
  "Bash(ls:*)" "Bash(cat:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(grep:*)" "Bash(jq:*)"
)

prompt() {
  cat <<EOF
Run exactly one experiment end to end and land it, following the land-experiment skill
(which builds on choose-next-experiment), then stop. You are in an unattended loop: don't ask questions; make the call the skill
describes and note it in the PR description.

- Plan with: uv run open-interpretability next --top 5 --epsilon $EPSILON --steps $CORPUS_OPTION$STEP_MODELS
  and follow the steps it prints.
- Proposer: pass --agent-model $PROPOSER to every \`open-interpretability audit\`.
- Reviewer: pass --agent-model $REVIEWER to every \`open-interpretability verify\`, replacing any
  <model …> placeholder in the planner's commands. If the draw needs a reviewer
  different from the record's proposer and $REVIEWER is the same model, re-draw once
  as the skill allows and say why in the PR.
- Corpus: use the corpus that each printed step names. Don't replace it. A review of an
  existing record must read the corpus that the record scanned.
- If a model is external/<name>, use the two-step flow in the skill's "Proposer and
  reviewer without API keys" section, with the neuron-proposer, neuron-reviewer and
  explanation-judge subagents answering the prompts. Every review needs explanation
  scores. Record the model IDs they report.
- Post the result whatever the verdict, then land it as land-experiment describes: build,
  wait for green checks, and merge only if every merge condition holds; otherwise leave
  the PR open with a comment saying why. If you have to abandon it, say why in the PR and
  land it with only its ledger entry (wiki/ledger/). Don't close it.
- If \`open-interpretability next\` says new experiments are paused, stop at once: claim nothing,
  and end with RESULT: none paused.
- Finish on a clean working tree. End your reply with one line:
  RESULT: <slug> <verdict> <PR url> <merged|open>   (or RESULT: none <reason>)
EOF
}

claude_args=(-p --output-format json --permission-mode acceptEdits --max-turns "$MAX_TURNS")
[[ -n $DRIVER ]] && claude_args+=(--model "$DRIVER")
claude_args+=(--allowedTools "${ALLOWED_TOOLS[@]}") # variadic, so last

# Wait while `open-interpretability pause` is in force (exit 75); any other failure doesn't wait.
wait_while_paused() {
  local said="" code
  while true; do
    code=0
    (cd "$WORKTREE" && uv run --quiet open-interpretability pause --status >/dev/null 2>&1) || code=$?
    [[ $code -ne 75 ]] && return 0
    stop_requested && return 0 # check_limits stops the loop
    if [[ -z $said ]]; then
      echo "[$1] $(cd "$WORKTREE" && uv run --quiet open-interpretability pause --status 2>/dev/null)"
      said=1
    fi
    sleep 60
  done
}

# The kill switch: while this file exists, no loop on the machine starts an experiment.
stop_file() { echo "${OPENINTERP_STOP_FILE:-${XDG_CACHE_HOME:-$HOME/.cache}/openinterp/stop}"; }
stop_requested() { [[ -e $(stop_file) ]]; }

# True when the cap $2 is set and the dollars spent, $1, reach it.
over_budget() { [[ -n $2 ]] && awk -v spent="$1" -v cap="$2" 'BEGIN { exit !(spent >= cap) }'; }

# Dollars reported by the ledgers and --session files given; 0 if none can be read.
spent() {
  local out
  out=$(cd "$WORKTREE" && uv run --quiet open-interpretability costs --usd "$@" 2>/dev/null) || out=0
  echo "${out:-0}"
}

# Every loop's ledgers and sessions logged today (UTC), as `open-interpretability costs` arguments.
todays_logs() {
  local f day
  day=$(date -u +%Y%m%d)
  for f in "$LOG_DIR/$day"T*.costs.jsonl; do [[ -e $f ]] && echo "$f"; done
  for f in "$LOG_DIR/$day"T*-[0-9]*.json; do [[ -e $f ]] && printf '%s\n' --session "$f"; done
}

# Stop between experiments when the stop file exists or a budget is spent.
check_limits() { # $1: the iteration about to start, or that just ended
  if stop_requested; then
    echo "[$1] stop file $(stop_file) exists; stopping. Remove it to run loops again."
    exit 0
  fi
  local args=() s f total today
  if [[ -n $BUDGET && ${#LEDGERS[@]} -gt 0 ]]; then
    args=("${LEDGERS[@]}")
    for s in "${SESSIONS[@]}"; do args+=(--session "$s"); done
    total=$(spent "${args[@]}")
    if over_budget "$total" "$BUDGET"; then
      echo "[$1] this loop spent \$$total, at or over --budget \$$BUDGET; stopping."
      exit 0
    fi
  fi
  if [[ -n $DAILY_BUDGET ]]; then
    args=()
    while IFS= read -r f; do args+=("$f"); done < <(todays_logs) # bash 3.2 has no mapfile
    today=0
    [[ ${#args[@]} -gt 0 ]] && today=$(spent "${args[@]}")
    if over_budget "$today" "$DAILY_BUDGET"; then
      echo "[$1] loops spent \$$today today (UTC), at or over --daily-budget \$$DAILY_BUDGET; stopping."
      exit 0
    fi
  fi
}

# `open-interpretability costs` from the worktree's checkout, which has the latest harness.
costs() { (cd "$WORKTREE" && uv run --quiet open-interpretability costs "$@") || true; }

# The session's final message, from its JSON result.
session_result() {
  (cd "$WORKTREE" && uv run --quiet python -c \
    'import json, sys; print(json.load(open(sys.argv[1])).get("result", ""))' "$1") 2>/dev/null || true
}

LEDGERS=()
SESSIONS=()
report() { # $1: iteration, $2: its ledger, $3: its session result
  echo "[$1] cost of this experiment:"
  costs "$2" --session "$3" | sed "s/^/[$1]   /"
  if [[ ${#LEDGERS[@]} -gt 1 ]]; then
    local args=("${LEDGERS[@]}") s
    for s in "${SESSIONS[@]}"; do args+=(--session "$s"); done
    echo "[$1] running total over ${#LEDGERS[@]} experiments:"
    costs "${args[@]}" | sed "s/^/[$1]   /"
  fi
}

i=0
while [[ $MAX -eq 0 || $i -lt $MAX ]]; do
  i=$((i + 1))
  check_limits "$i"
  stem="$LOG_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$i"
  log="$stem.log" ledger="$stem.costs.jsonl" session="$stem.json"
  LEDGERS+=("$ledger")
  SESSIONS+=("$session")
  echo "[$i] $(date -u +%FT%TZ) starting; log: $log"

  # Every experiment starts from the latest registry, detached so no branch is reused.
  git -C "$WORKTREE" fetch --quiet origin main
  if [[ -n $(git -C "$WORKTREE" status --porcelain) ]]; then
    echo "[$i] worktree $WORKTREE is dirty; stopping so nothing is lost" >&2
    exit 1
  fi
  git -C "$WORKTREE" switch --quiet --detach origin/main
  # Build the venv now, so its output doesn't bury the session's first `open-interpretability next`.
  (cd "$WORKTREE" && uv sync --quiet) || true
  wait_while_paused "$i"
  check_limits "$i"

  status=0
  (cd "$WORKTREE" && prompt | OPENINTERP_COSTS_LOG="$ledger" "$CLAUDE_BIN" "${claude_args[@]}") \
    >"$session" 2>"$log" || status=$?
  session_result "$session" >>"$log"
  report "$i" "$ledger" "$session"
  if [[ $status -ne 0 ]]; then
    echo "[$i] claude exited with an error (usage limit?); stopping. See $log" >&2
    exit 1
  fi
  grep -E '^RESULT:' "$log" | tail -n 1 | sed "s/^/[$i] /" || echo "[$i] done (no RESULT line)"
  [[ -n $BUDGET || -n $DAILY_BUDGET ]] && (cd "$WORKTREE" && uv run --quiet open-interpretability costs --usd \
    "$ledger" --session "$session" 2>&1 >/dev/null | sed "s/^/[$i] warning: budget skips /") || true
done
echo "Reached --max $MAX; stopping."
