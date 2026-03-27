#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_eval_sealqa_local_openai.sh
# Or override defaults via env vars, e.g.:
#   MODEL=deepseek-v3.2 NUM_SAMPLES=2 bash scripts/run_eval_sealqa_local_openai.sh

OPENAI_BASE_URL_DEFAULT="http://s-20260304094348-lphmr.ailab-pj.pjh-service.org.cn/v1"
MODEL_DEFAULT="deepseek-v3.2"
DATASET_DEFAULT=".dataset/seal-0.csv"
OUTPUT_DEFAULT="results/sealqa_eval_local_openai.pkl"
MAX_CONCURRENT_DEFAULT="1"

OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_BASE_URL_DEFAULT}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
MODEL="${MODEL:-$MODEL_DEFAULT}"
DATASET="${DATASET:-$DATASET_DEFAULT}"
OUTPUT="${OUTPUT:-$OUTPUT_DEFAULT}"
MAX_CONCURRENT="${MAX_CONCURRENT:-$MAX_CONCURRENT_DEFAULT}"

ARGS=(
  --sdk openai
  --task sealqa
  --model "$MODEL"
  --dataset "$DATASET"
  --output "$OUTPUT"
  --max-concurrent "$MAX_CONCURRENT"
)

# By default, evaluate all samples. Set NUM_SAMPLES to limit.
if [[ -n "${NUM_SAMPLES:-}" ]]; then
  ARGS+=(--num-samples "$NUM_SAMPLES")
fi

OPENAI_BASE_URL="$OPENAI_BASE_URL" \
OPENAI_API_KEY="$OPENAI_API_KEY" \
python scripts/run_eval_runner.py "${ARGS[@]}"
