#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Control group (DabStep): no loop, direct eval with isolated empty skills profile.

EXP_ROOT=${EXP_ROOT:-experiments/repro_3bench}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
TASK_DIR="$EXP_ROOT/$RUN_TAG/dabstep/control_no_iter"
SKILLS_DIR="$TASK_DIR/skills_profile"
RESULT_PKL="$TASK_DIR/eval/results.pkl"
SUMMARY_JSON="$TASK_DIR/eval/summary.json"
AGG_CSV="$EXP_ROOT/$RUN_TAG/summary/all_runs.csv"

mkdir -p "$TASK_DIR/eval" "$SKILLS_DIR"

DEBUG_EVAL=${DEBUG_EVAL:-1}
EVAL_HEARTBEAT_SEC=${EVAL_HEARTBEAT_SEC:-30}
AGENT_TIMEOUT_SEC=${AGENT_TIMEOUT_SEC:-1200}
AGENT_MAX_RETRIES=${AGENT_MAX_RETRIES:-3}
AGENT_INITIAL_BACKOFF_SEC=${AGENT_INITIAL_BACKOFF_SEC:-30}
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-1800}

export EVOSKILL_EVAL_DEBUG="$DEBUG_EVAL"
export EVOSKILL_EVAL_HEARTBEAT_SEC="$EVAL_HEARTBEAT_SEC"
export EVOSKILL_AGENT_TIMEOUT_SEC="$AGENT_TIMEOUT_SEC"
export EVOSKILL_AGENT_MAX_RETRIES="$AGENT_MAX_RETRIES"
export EVOSKILL_AGENT_INITIAL_BACKOFF_SEC="$AGENT_INITIAL_BACKOFF_SEC"
export EVOSKILL_EVAL_TIMEOUT_SEC="$EVAL_TIMEOUT_SEC"

echo "Debug config: EVOSKILL_EVAL_DEBUG=$EVOSKILL_EVAL_DEBUG, EVOSKILL_EVAL_HEARTBEAT_SEC=$EVOSKILL_EVAL_HEARTBEAT_SEC"
echo "Timeout config: AGENT=${EVOSKILL_AGENT_TIMEOUT_SEC}s x${EVOSKILL_AGENT_MAX_RETRIES}, EVAL=${EVOSKILL_EVAL_TIMEOUT_SEC}s, BACKOFF=${EVOSKILL_AGENT_INITIAL_BACKOFF_SEC}s"

COMMON_OPENAI_ARGS=()
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  COMMON_OPENAI_ARGS+=(--openai-base-url "$OPENAI_BASE_URL")
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  COMMON_OPENAI_ARGS+=(--openai-api-key "$OPENAI_API_KEY")
fi

python scripts/run_eval_dabstep.py \
  --dataset .dataset/dabstep_data.csv \
  --output "$RESULT_PKL" \
  --max-concurrent ${EVAL_MAX_CONCURRENT:-8} \
  --no-resume \
  --skills-dir "$SKILLS_DIR" \
  --sdk ${SDK:-claude} \
  --model ${MODEL:-claude-opus-4-5-20251101} \
  "${COMMON_OPENAI_ARGS[@]}"

python scripts/repro_templates/common/summarize_results.py \
  --task dabstep \
  --result-pkl "$RESULT_PKL" \
  --summary-json "$SUMMARY_JSON" \
  --append-csv "$AGG_CSV"
