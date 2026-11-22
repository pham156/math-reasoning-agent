#!/usr/bin/env bash
#SBATCH --job-name=sft_pilot
#SBATCH --account=scholar
#SBATCH --partition=gpu
##SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}}"
cd "$PROJECT_ROOT"
source scripts/setup_env.sh >/dev/null

ROOT="${SFT_STUDY_ROOT:-$OUTPUT_DIR/rl_1p5b_study/sft_pilot}"
mkdir -p "$ROOT" "$LOG_DIR"

echo "=== Running SFT Pilot ==="
echo "Node: $SLURMD_NODENAME | GPU: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

export RL_SMOKE_GPU=1

bash scripts/rl/smoke_python.sh -m src.training.rl.sft_pilot \
  --output_dir "$ROOT"
