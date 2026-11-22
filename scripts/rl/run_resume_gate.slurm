#!/usr/bin/env bash
#SBATCH --job-name=rl_gate
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
GATE_DIR="$ROOT/resume_gate/$METHOD"
mkdir -p "$GATE_DIR" "$LOG_DIR"

echo "=== Running Resume Gate for $METHOD (task $SLURM_ARRAY_TASK_ID) ==="
echo "Node: $SLURMD_NODENAME | GPU: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

export RL_SMOKE_GPU=1

# 1. Run 8 steps uninterrupted
bash scripts/rl/smoke_python.sh -m src.training.rl.pilot \
  --algorithm "$METHOD" \
  --output_dir "$GATE_DIR/continuous" \
  --max_steps 8 \
  --stop_after_step 8 \
  --save_steps 4 \
  --eval_samples 2

# 2. Run first 4 steps of resumed run
bash scripts/rl/smoke_python.sh -m src.training.rl.pilot \
  --algorithm "$METHOD" \
  --output_dir "$GATE_DIR/resumed" \
  --max_steps 8 \
  --stop_after_step 4 \
  --save_steps 4 \
  --eval_samples 2

# 3. Check boundary at step 4
bash scripts/rl/smoke_python.sh -m src.evaluation.check_rl_resume "$GATE_DIR" --boundary_only

# 4. Resume from step 4 to step 8
bash scripts/rl/smoke_python.sh -m src.training.rl.pilot \
  --algorithm "$METHOD" \
  --output_dir "$GATE_DIR/resumed" \
  --max_steps 8 \
  --stop_after_step 8 \
  --resume \
  --save_steps 4 \
  --eval_samples 2

# 5. Check full 8-step equivalence
bash scripts/rl/smoke_python.sh -m src.evaluation.check_rl_resume "$GATE_DIR"
echo "Resume gate passed for $METHOD"
