#!/bin/bash
#SBATCH --job-name=math-agent-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/pham156/math-reasoning-agent/logs/%x-%j.out
#SBATCH --error=/home/pham156/math-reasoning-agent/logs/%x-%j.err

set -euo pipefail

cd /home/pham156/math-reasoning-agent

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate math-reasoning-agent || true
elif [[ -f env/math-reasoning-agent/bin/activate ]]; then
  source env/math-reasoning-agent/bin/activate
fi

source scripts/setup_env.sh

python src/evaluate_agent.py \
  --split "${SPLIT:-test}" \
  --limit "${LIMIT:-20}" \
  --modes "${MODES:-baseline,retrieval,verifier,search}" \
  --experiment_name "${EXPERIMENT_NAME:-slurm_eval_${SLURM_JOB_ID}}" \
  --top_k "${TOP_K:-3}" \
  --candidate_k "${CANDIDATE_K:-30}" \
  --max_iterations "${MAX_ITERATIONS:-1}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-700}" \
  --max_depth "${MAX_DEPTH:-2}" \
  --beam_width "${BEAM_WIDTH:-3}" \
  --branch_factor "${BRANCH_FACTOR:-2}"
