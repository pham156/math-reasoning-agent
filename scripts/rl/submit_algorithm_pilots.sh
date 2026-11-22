#!/usr/bin/env bash
# Dry-run by default; explicitly pass --submit to queue the staged pilot matrix.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}}"
cd "$PROJECT_ROOT"
source scripts/setup_env.sh >/dev/null
MODE="${1:---dry-run}"
if [[ "$MODE" != --dry-run && "$MODE" != --submit ]]; then
  echo 'Usage: bash scripts/rl/submit_algorithm_pilots.sh [--dry-run|--submit]' >&2
  exit 2
fi

SBATCH_OPTS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SBATCH_OPTS+=(--account="$SLURM_ACCOUNT")
fi
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_OPTS+=(--partition="$SLURM_PARTITION")
fi
if [[ -n "${SLURM_CONSTRAINT+x}" ]]; then
  SBATCH_OPTS+=(--constraint="$SLURM_CONSTRAINT")
fi
if [[ -n "${SLURM_QOS:-}" ]]; then
  SBATCH_OPTS+=(--qos="$SLURM_QOS")
fi
if [[ -n "${SLURM_TIME:-}" ]]; then
  SBATCH_OPTS+=(--time="$SLURM_TIME")
fi

export RL_STUDY_ROOT="${RL_STUDY_ROOT:-$OUTPUT_DIR/rl_1p5b_study/pilot_$(date +%Y%m%d_%H%M%S)}"
if [[ "$RL_STUDY_ROOT" != "$OUTPUT_DIR/"* ]]; then
  echo 'Study root must be inside scratch OUTPUT_DIR' >&2
  exit 2
fi
echo "Study: $RL_STUDY_ROOT"
echo 'Four algorithms, seed 0, correctness-only reward, 512 shared train / 20 reserved development diagnostics.'
echo 'Four resume gates first; then up to three 4-hour training segments per algorithm. Concurrency: 2.'
echo 'Full 5,000-example and reward-design grids are not included.'
for SCRIPT in scripts/rl/run_resume_gate.slurm scripts/rl/run_algorithm_pilots.slurm scripts/rl/summarize_algorithm_pilots.slurm; do
  bash -n "$SCRIPT"
  sbatch "${SBATCH_OPTS[@]}" --test-only "$SCRIPT"
done
if [[ "$MODE" == --dry-run ]]; then
  exit 0
fi
if [[ ! -f "$DATA_DIR/rl_1p5b_pilot_v1/manifest.json" ]]; then
  echo 'Missing frozen split; run the documented --prepare_splits command first' >&2
  exit 2
fi
mkdir -p "$RL_STUDY_ROOT"
exec > >(tee -a "$RL_STUDY_ROOT/submission.log") 2>&1
echo "Study: $RL_STUDY_ROOT"
GATE=$(sbatch "${SBATCH_OPTS[@]}" --parsable scripts/rl/run_resume_gate.slurm)
GATE="${GATE%%;*}"
echo "Resume gate: $GATE"
PARENT="$GATE"
for SEGMENT in 1 2 3; do
  JOB=$(sbatch "${SBATCH_OPTS[@]}" --parsable --kill-on-invalid-dep=yes --dependency="afterok:$PARENT" scripts/rl/run_algorithm_pilots.slurm)
  JOB="${JOB%%;*}"
  echo "Pilot segment $SEGMENT: $JOB (afterok:$PARENT)"
  PARENT="$JOB"
done
SUMMARY=$(sbatch "${SBATCH_OPTS[@]}" --parsable --dependency="afterany:$PARENT" scripts/rl/summarize_algorithm_pilots.slurm)
echo "Summary: ${SUMMARY%%;*} (afterany:$PARENT)"
