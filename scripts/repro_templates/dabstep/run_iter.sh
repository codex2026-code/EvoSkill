#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Iterative group (DabStep): EvoSkill API loop -> run_eval_dabstep.py

EXP_ROOT=${EXP_ROOT:-experiments/repro_3bench}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
TASK_DIR="$EXP_ROOT/$RUN_TAG/dabstep/iter_skill"
SKILLS_DIR="$TASK_DIR/skills_profile"
RESULT_PKL="$TASK_DIR/eval/results.pkl"
SUMMARY_JSON="$TASK_DIR/eval/summary.json"
AGG_CSV="$EXP_ROOT/$RUN_TAG/summary/all_runs.csv"

mkdir -p "$TASK_DIR/eval" "$SKILLS_DIR"

if [[ "${RUN_DIAG:-0}" == "1" ]]; then
  export EVOSKILL_EVAL_DEBUG="${EVOSKILL_EVAL_DEBUG:-1}"
  export EVOSKILL_EVAL_HEARTBEAT="${EVOSKILL_EVAL_HEARTBEAT:-1}"
  export EVOSKILL_EVAL_HEARTBEAT_SEC="${EVOSKILL_EVAL_HEARTBEAT_SEC:-10}"
  export EVOSKILL_AGENT_TIMEOUT_SEC="${EVOSKILL_AGENT_TIMEOUT_SEC:-300}"
  export EVOSKILL_EVAL_TIMEOUT_SEC="${EVOSKILL_EVAL_TIMEOUT_SEC:-600}"
  echo "[DIAG] RUN_DIAG=1 enabled with defaults:"
  echo "       EVOSKILL_EVAL_DEBUG=$EVOSKILL_EVAL_DEBUG"
  echo "       EVOSKILL_EVAL_HEARTBEAT=$EVOSKILL_EVAL_HEARTBEAT"
  echo "       EVOSKILL_EVAL_HEARTBEAT_SEC=$EVOSKILL_EVAL_HEARTBEAT_SEC"
  echo "       EVOSKILL_AGENT_TIMEOUT_SEC=$EVOSKILL_AGENT_TIMEOUT_SEC"
  echo "       EVOSKILL_EVAL_TIMEOUT_SEC=$EVOSKILL_EVAL_TIMEOUT_SEC"
fi

COMMON_OPENAI_ARGS=()
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  COMMON_OPENAI_ARGS+=(--openai-base-url "$OPENAI_BASE_URL")
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  COMMON_OPENAI_ARGS+=(--openai-api-key "$OPENAI_API_KEY")
fi

python scripts/repro_templates/common/run_loop_evoskill.py \
  --task dabstep \
  --dataset .dataset/dabstep_data.csv \
  --mode skill_only \
  --max-iterations ${MAX_ITERATIONS:-20} \
  --frontier-size ${FRONTIER_SIZE:-3} \
  --concurrency ${CONCURRENCY:-4} \
  --failure-samples ${FAILURE_SAMPLES:-3} \
  --skills-dir "$SKILLS_DIR" \
  --loop-summary-json "$TASK_DIR/loop/iteration_log.json" \
  --sdk ${SDK:-claude} \
  --model ${MODEL:-claude-opus-4-5-20251101} \
  "${COMMON_OPENAI_ARGS[@]}"

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
