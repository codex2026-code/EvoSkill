#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Iterative group (SEAL-QA): run_loop_sealqa.py -> run_eval_sealqa.py
# Keep this path layout unchanged for reproducibility.

EXP_ROOT=${EXP_ROOT:-experiments/repro_3bench}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
TASK_DIR="$EXP_ROOT/$RUN_TAG/sealqa/iter_skill"
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

GRADER_ARGS=(--grader-model "${GRADER_MODEL:-openai/gpt-5-mini}")
if [[ -n "${GRADER_BASE_URL:-}" ]]; then
  GRADER_ARGS+=(--grader-base-url "$GRADER_BASE_URL")
elif [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  GRADER_ARGS+=(--grader-base-url "$OPENAI_BASE_URL")
fi
if [[ -n "${GRADER_API_KEY:-}" ]]; then
  GRADER_ARGS+=(--grader-api-key "$GRADER_API_KEY")
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  GRADER_ARGS+=(--grader-api-key "$OPENAI_API_KEY")
fi

python scripts/run_loop_sealqa.py \
  --mode skill_only \
  --dataset .dataset/seal-0.csv \
  --max-iterations ${MAX_ITERATIONS:-20} \
  --frontier-size ${FRONTIER_SIZE:-3} \
  --concurrency ${CONCURRENCY:-4} \
  --failure-samples ${FAILURE_SAMPLES:-3} \
  --skills-dir "$SKILLS_DIR" \
  --iteration-log-json "$TASK_DIR/loop/iteration_log.json" \
  --debug-eval \
  --eval-heartbeat-sec ${EVAL_HEARTBEAT_SEC} \
  --sdk ${SDK:-claude} \
  --model ${MODEL:-claude-opus-4-5-20251101} \
  "${COMMON_OPENAI_ARGS[@]}" \
  "${GRADER_ARGS[@]}"

python scripts/run_eval_sealqa.py \
  --dataset .dataset/seal-0.csv \
  --output "$RESULT_PKL" \
  --max-concurrent ${EVAL_MAX_CONCURRENT:-8} \
  --no-resume \
  --skills-dir "$SKILLS_DIR" \
  --debug-eval \
  --eval-heartbeat-sec ${EVAL_HEARTBEAT_SEC} \
  --sdk ${SDK:-claude} \
  --model ${MODEL:-claude-opus-4-5-20251101} \
  "${COMMON_OPENAI_ARGS[@]}" \
  "${GRADER_ARGS[@]}"

python scripts/repro_templates/common/summarize_results.py \
  --task sealqa \
  --result-pkl "$RESULT_PKL" \
  --summary-json "$SUMMARY_JSON" \
  --append-csv "$AGG_CSV" \
  "${GRADER_ARGS[@]}"
