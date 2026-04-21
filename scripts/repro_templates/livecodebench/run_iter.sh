#!/usr/bin/env bash
set -euo pipefail

# Iterative group (LiveCodeBench): EvoSkill API loop -> run_eval_livecodebench.py

EXP_ROOT=${EXP_ROOT:-experiments/repro_3bench}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
TASK_DIR="$EXP_ROOT/$RUN_TAG/livecodebench/iter_skill"
SKILLS_DIR="$TASK_DIR/skills_profile"
RESULT_PKL="$TASK_DIR/eval/results.pkl"
SUMMARY_JSON="$TASK_DIR/eval/summary.json"
AGG_CSV="$EXP_ROOT/$RUN_TAG/summary/all_runs.csv"

mkdir -p "$TASK_DIR/eval" "$SKILLS_DIR"

python scripts/repro_templates/common/run_loop_evoskill.py \
  --task livecodebench \
  --mode skill_only \
  --max-iterations ${MAX_ITERATIONS:-20} \
  --frontier-size ${FRONTIER_SIZE:-3} \
  --concurrency ${CONCURRENCY:-4} \
  --failure-samples ${FAILURE_SAMPLES:-3} \
  --skills-dir "$SKILLS_DIR" \
  --loop-summary-json "$TASK_DIR/loop/iteration_log.json" \
  --sdk ${SDK:-claude} \
  --model ${MODEL:-claude-opus-4-5-20251101}

python scripts/run_eval_livecodebench.py \
  --output "$RESULT_PKL" \
  --max-concurrent ${EVAL_MAX_CONCURRENT:-4} \
  --no-resume \
  --skills-dir "$SKILLS_DIR" \
  --sdk ${SDK:-claude} \
  --model ${MODEL:-claude-opus-4-5-20251101}

python scripts/repro_templates/common/summarize_results.py \
  --task livecodebench \
  --result-pkl "$RESULT_PKL" \
  --summary-json "$SUMMARY_JSON" \
  --append-csv "$AGG_CSV"

