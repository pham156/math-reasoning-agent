#!/usr/bin/env bash
#SBATCH --job-name=rl_pilot
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}}"
cd "$PROJECT_ROOT"
source scripts/setup_env.sh >/dev/null

ALGORITHMS=(grpo dr_grpo dapo_style gspo)
METHOD="${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}"
ROOT="${RL_STUDY_ROOT:-$OUTPUT_DIR/rl_1p5b_study/latest}"
RUN_DIR="$ROOT/pilot/$METHOD"
mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "=== Running Pilot Segment for $METHOD (task $SLURM_ARRAY_TASK_ID) ==="
echo "Node: $SLURMD_NODENAME | GPU: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

export RL_SMOKE_GPU=1

bash scripts/rl/smoke_python.sh -m src.training.rl.pilot \
  --algorithm "$METHOD" \
  --output_dir "$RUN_DIR" \
  --max_steps 128 \
  --segment_seconds 9000 \
  --eval_samples 20 \
  --resume
