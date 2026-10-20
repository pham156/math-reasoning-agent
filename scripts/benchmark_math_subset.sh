#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/setup_env.sh"

MODES=${MODES:-baseline,retrieval,verifier,search}
LIMIT=${LIMIT:-20}
SPLIT=${SPLIT:-test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-math_subset_$(date +%Y%m%d_%H%M%S)}

python -m src.evaluation.evaluate_agent \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --modes "$MODES" \
  --experiment_name "$EXPERIMENT_NAME" \
  --top_k "${TOP_K:-3}" \
  --candidate_k "${CANDIDATE_K:-30}" \
  --max_iterations "${MAX_ITERATIONS:-1}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-700}" \
  --max_depth "${MAX_DEPTH:-2}" \
  --beam_width "${BEAM_WIDTH:-3}" \
  --branch_factor "${BRANCH_FACTOR:-2}" \
  "$@"
