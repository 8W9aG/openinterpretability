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

  --proposer MODEL     LiteLLM model or external/<name> for `openinterp audit`
                       (default: $OPENINTERP_AGENT_MODEL, else anthropic/claude-opus-5-5)
  --reviewer MODEL     LiteLLM model or external/<name> for `openinterp verify`
                       (default: same as the proposer; records are then not independent)
  --driver MODEL       Claude model that drives each session (default: claude's own)
  --corpus PATH        Scan corpus (default: corpus.jsonl in the repo root)
  --max N              Stop after N experiments; 0 = until claude fails (default: 0)
  --epsilon P          Exploration probability passed to `openinterp next` (default: 0.7)
  --worktree DIR       Worktree to run in (default: ../<repo>-loop, created if missing)
  --max-turns N        Turn limit per session, a guard against runaway sessions (default: 200)
  -h, --help           Show this help

Logs go to <repo>/logs/loop/: per experiment, the session's stderr and result (.log),
its `claude --output-format json` result (.json), and the costs ledger every openinterp
command appends to (.costs.jsonl). After each experiment the loop prints what it cost
and a running total (`openinterp costs`). Set CLAUDE_BIN if `claude` is not on PATH.
EOF
}

REPO=$(git rev-parse --show-toplevel)
PROPOSER=${OPENINTERP_AGENT_MODEL:-anthropic/claude-opus-5-5}
REVIEWER=""
DRIVER=""
CORPUS="$REPO/corpus.jsonl"
MAX=0
EPSILON=0.7
WORKTREE="$(dirname "$REPO")/$(basename "$REPO")-loop"
MAX_TURNS=200
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
[[ -f $CORPUS ]] || { echo "corpus not found: $CORPUS (see README, step 1)" >&2; exit 1; }
CORPUS=$(cd "$(dirname "$CORPUS")" && pwd)/$(basename "$CORPUS")
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

- Plan with: uv run openinterp next --top 5 --epsilon $EPSILON --steps --corpus $CORPUS$STEP_MODELS
  and follow the steps it prints.
- Proposer: pass --agent-model $PROPOSER to every \`openinterp audit\`.
- Reviewer: pass --agent-model $REVIEWER to every \`openinterp verify\`, replacing any
  <model …> placeholder in the planner's commands. If the draw needs a reviewer
  different from the record's proposer and $REVIEWER is the same model, re-draw once
  as the skill allows and say why in the PR.
- Corpus: add --corpus $CORPUS to every \`openinterp audit\`, \`openinterp score\` and
  \`openinterp verify\` (verify scores the explanation on held-out text from it).
- If a model is external/<name>, use the two-step flow in the skill's "Proposer and
  reviewer without API keys" section, with the neuron-proposer, neuron-reviewer and
  explanation-judge subagents answering the prompts. Every review needs explanation
  scores. Record the model IDs they report.
- Post the result whatever the verdict, then land it as land-experiment describes: build,
  wait for green checks, and merge only if every merge condition holds; otherwise leave
  the PR open with a comment saying why. If you have to abandon it, close the PR with
  the reason.
- Finish on a clean working tree. End your reply with one line:
  RESULT: <slug> <verdict> <PR url> <merged|open>   (or RESULT: none <reason>)
EOF
}

claude_args=(-p --output-format json --permission-mode acceptEdits --max-turns "$MAX_TURNS")
[[ -n $DRIVER ]] && claude_args+=(--model "$DRIVER")
claude_args+=(--allowedTools "${ALLOWED_TOOLS[@]}") # variadic, so last

# `openinterp costs` from the worktree's checkout, which has the latest harness.
costs() { (cd "$WORKTREE" && uv run --quiet openinterp costs "$@") || true; }

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
  # Build the venv now, so its output doesn't bury the session's first `openinterp next`.
  (cd "$WORKTREE" && uv sync --quiet) || true

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
done
echo "Reached --max $MAX; stopping."
