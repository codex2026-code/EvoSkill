#!/usr/bin/env bash
set -euo pipefail

# One-click template runner for 3 datasets x 2 configs.
# You may set RUN_TAG explicitly to reuse one experiment folder.

RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
export RUN_TAG

bash scripts/repro_templates/sealqa/run_control.sh
bash scripts/repro_templates/sealqa/run_iter.sh

bash scripts/repro_templates/dabstep/run_control.sh
bash scripts/repro_templates/dabstep/run_iter.sh

bash scripts/repro_templates/livecodebench/run_control.sh
bash scripts/repro_templates/livecodebench/run_iter.sh

echo "Done. Aggregated CSV: ${EXP_ROOT:-experiments/repro_3bench}/$RUN_TAG/summary/all_runs.csv"
