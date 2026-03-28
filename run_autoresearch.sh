#!/usr/bin/env bash
set -euo pipefail

CODEX_BIN=${CODEX_BIN:-codex}
LOGFILE=${LOGFILE:-agent-loop.log}
NORMAL_SLEEP=${NORMAL_SLEEP:-2}
ERROR_SLEEP=${ERROR_SLEEP:-30}
DEFAULT_PROMPT="Read program.md and continue the autonomous experiment loop from the current best state. Do not stop after a summary, a revert, or a new best result. Immediately start the next experiment. Explore beyond hyperparameter tuning. If a candidate fails to produce a real running process or only rewrites the startup line to run.log, treat it as an infrastructure crash, log it, restore the previous best state, and move on to a different experiment. Do not retry the same stuck candidate indefinitely."
PROMPT="$DEFAULT_PROMPT"
FIRST_ADDENDUM=""

usage() {
  echo "Usage: $0 [prompt] [--first-addendum TEXT]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --first-addendum)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      FIRST_ADDENDUM=$2
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option '$1'." >&2
      usage
      exit 1
      ;;
    *)
      if [[ "$PROMPT" != "$DEFAULT_PROMPT" ]]; then
        echo "Error: only one positional prompt is supported." >&2
        usage
        exit 1
      fi
      PROMPT=$1
      shift
      ;;
  esac
done

if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "Error: '$CODEX_BIN' not found on PATH." >&2
  exit 1
fi

while true; do
  RUN_PROMPT=$PROMPT
  if [[ -n "$FIRST_ADDENDUM" ]]; then
    RUN_PROMPT="${PROMPT}"$'\n\n'"One-time addendum for this first run only: ${FIRST_ADDENDUM}"
  fi

  echo
  echo "[$(date -Is)] starting fresh codex run" | tee -a "$LOGFILE"

  set +e
  "$CODEX_BIN" exec \
    --dangerously-bypass-approvals-and-sandbox \
    -c model_reasoning_effort="medium" \
    "$RUN_PROMPT" 2>&1 | tee -a "$LOGFILE"
  status=${PIPESTATUS[0]}
  set -e

  FIRST_ADDENDUM=""

  if tail -n 80 "$LOGFILE" | grep -q "failed to connect to websocket\|500 Internal Server Error\|401 Unauthorized"; then
    echo "[$(date -Is)] api/auth failure; sleeping ${ERROR_SLEEP}s" | tee -a "$LOGFILE"
    sleep "$ERROR_SLEEP"
  else
    echo "[$(date -Is)] codex turn ended with status ${status}; sleeping ${NORMAL_SLEEP}s" | tee -a "$LOGFILE"
    sleep "$NORMAL_SLEEP"
  fi
done