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
CLEAN_BEFORE_RUN=${CLEAN_BEFORE_RUN:-1}
CACHE_DIR=${CACHE_DIR:-.cache/runs}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-artifacts/dabstep}

cleanup_previous_outputs() {
  [[ "$CLEAN_BEFORE_RUN" == "1" ]] || return 0
  echo "[cleanup] CLEAN_BEFORE_RUN=1, removing stale outputs for RUN_TAG=$RUN_TAG"

  # Remove prior run outputs for the same tag (loop/eval/skills profile snapshots).
  rm -rf "$TASK_DIR"

  # Remove cached traces to avoid stale hit/mismatch against updated programs or prompts.
  rm -rf "$CACHE_DIR"

  # Remove DabStep artifact registry/workspace snapshots so base program state starts fresh.
  rm -rf "$ARTIFACTS_DIR"
}

cleanup_previous_outputs

mkdir -p "$TASK_DIR/eval" "$SKILLS_DIR"

# Optional memory guard for iterative DabStep workflow.
# Disabled by default to preserve original behavior unless explicitly enabled.
MEM_GUARD_ENABLED=${MEM_GUARD_ENABLED:-0}
MEM_GUARD_MAX_RATIO=${MEM_GUARD_MAX_RATIO:-0}
MEM_GUARD_RESUME_RATIO=${MEM_GUARD_RESUME_RATIO:-}
MEM_GUARD_CHECK_INTERVAL_SEC=${MEM_GUARD_CHECK_INTERVAL_SEC:-2}
MEM_GUARD_LOG_PREFIX=${MEM_GUARD_LOG_PREFIX:-[dabstep-iter-mem-guard]}

if [[ -z "$MEM_GUARD_RESUME_RATIO" ]]; then
  MEM_GUARD_RESUME_RATIO="$MEM_GUARD_MAX_RATIO"
fi

_float_lt() {
  awk -v lhs="$1" -v rhs="$2" 'BEGIN { exit !(lhs < rhs) }'
}

_float_gt() {
  awk -v lhs="$1" -v rhs="$2" 'BEGIN { exit !(lhs > rhs) }'
}

_float_in_0_1() {
  awk -v val="$1" 'BEGIN { exit !(val >= 0 && val <= 1) }'
}

validate_mem_guard_config() {
  [[ "$MEM_GUARD_ENABLED" == "1" ]] || return 0
  if ! _float_in_0_1 "$MEM_GUARD_MAX_RATIO"; then
    echo "ERROR: MEM_GUARD_MAX_RATIO must be in [0,1], got: $MEM_GUARD_MAX_RATIO" >&2
    exit 1
  fi
  if ! _float_in_0_1 "$MEM_GUARD_RESUME_RATIO"; then
    echo "ERROR: MEM_GUARD_RESUME_RATIO must be in [0,1], got: $MEM_GUARD_RESUME_RATIO" >&2
    exit 1
  fi
  if _float_gt "$MEM_GUARD_RESUME_RATIO" "$MEM_GUARD_MAX_RATIO"; then
    echo "ERROR: MEM_GUARD_RESUME_RATIO must be <= MEM_GUARD_MAX_RATIO" >&2
    exit 1
  fi
}

run_with_memory_guard() {
  if [[ "$MEM_GUARD_ENABLED" != "1" || "$MEM_GUARD_MAX_RATIO" == "0" ]]; then
    "$@"
    return $?
  fi

  local -r total_kb
  total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  if [[ -z "$total_kb" || "$total_kb" -le 0 ]]; then
    echo "$MEM_GUARD_LOG_PREFIX unable to read MemTotal, fallback to unguarded run." >&2
    "$@"
    return $?
  fi

  "$@" &
  local child_pid=$!
  local is_stopped=0

  while kill -0 "$child_pid" 2>/dev/null; do
    local rss_kb
    rss_kb="$(awk '/^VmRSS:/ {print $2}' "/proc/$child_pid/status" 2>/dev/null || true)"
    if [[ -n "$rss_kb" ]]; then
      local ratio
      ratio="$(awk -v rss="$rss_kb" -v total="$total_kb" 'BEGIN { printf "%.6f", rss / total }')"
      if [[ "$is_stopped" -eq 0 ]] && _float_gt "$ratio" "$MEM_GUARD_MAX_RATIO"; then
        kill -STOP "$child_pid" 2>/dev/null || true
        is_stopped=1
        echo "$MEM_GUARD_LOG_PREFIX pid=$child_pid paused rss_ratio=$ratio > max=$MEM_GUARD_MAX_RATIO" >&2
      elif [[ "$is_stopped" -eq 1 ]] && _float_lt "$ratio" "$MEM_GUARD_RESUME_RATIO"; then
        kill -CONT "$child_pid" 2>/dev/null || true
        is_stopped=0
        echo "$MEM_GUARD_LOG_PREFIX pid=$child_pid resumed rss_ratio=$ratio < resume=$MEM_GUARD_RESUME_RATIO" >&2
      fi
    fi
    sleep "$MEM_GUARD_CHECK_INTERVAL_SEC"
  done

  wait "$child_pid"
}

validate_mem_guard_config

COMMON_OPENAI_ARGS=()
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  COMMON_OPENAI_ARGS+=(--openai-base-url "$OPENAI_BASE_URL")
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  COMMON_OPENAI_ARGS+=(--openai-api-key "$OPENAI_API_KEY")
fi

run_with_memory_guard python scripts/repro_templates/common/run_loop_evoskill.py \
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

run_with_memory_guard python scripts/run_eval_dabstep.py \
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
