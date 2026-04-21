#!/usr/bin/env bash
set -euo pipefail

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
  --model ${MODEL:-claude-opus-4-5-20251101}

python scripts/run_eval_sealqa.py \
  --dataset .dataset/seal-0.csv \
  --output "$RESULT_PKL" \
  --max-concurrent ${EVAL_MAX_CONCURRENT:-8} \
  --no-resume \
  --skills-dir "$SKILLS_DIR" \
  --sdk ${SDK:-claude} \
  --model ${MODEL:-claude-opus-4-5-20251101}

python scripts/repro_templates/common/summarize_results.py \
  --task sealqa \
  --result-pkl "$RESULT_PKL" \
  --summary-json "$SUMMARY_JSON" \
  --append-csv "$AGG_CSV" \
  --grader-model ${GRADER_MODEL:-openai/gpt-5-mini}

