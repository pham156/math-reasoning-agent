#!/usr/bin/env bash
#SBATCH --job-name=rl_eval
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}}"
cd "$PROJECT_ROOT"
source scripts/setup_env.sh >/dev/null

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
NAME="${NAME:-dev300_eval}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=== Evaluating $NAME: $MODEL_PATH ==="
echo "Node: $SLURMD_NODENAME | GPU: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

export RL_SMOKE_GPU=1

bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_dev \
  --model_path "$MODEL_PATH" \
  --name "$NAME" \
  --output_dir "$OUTPUT_DIR"
