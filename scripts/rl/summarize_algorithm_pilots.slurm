#!/usr/bin/env bash
#SBATCH --job-name=rl_summary
#SBATCH --account=scholar
#SBATCH --partition=cpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}}"
cd "$PROJECT_ROOT"
source scripts/setup_env.sh >/dev/null

ROOT="${RL_STUDY_ROOT:-$OUTPUT_DIR/rl_1p5b_study/latest}"
mkdir -p "$LOG_DIR"

echo "=== Summarizing Algorithm Pilots in $ROOT ==="
bash scripts/rl/smoke_python.sh -m src.evaluation.summarize_pilots "$ROOT"
