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
