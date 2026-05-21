#!/bin/bash
#SBATCH --job-name=math-agent-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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

python src/run_pipeline.py \
  --problem "${PROBLEM:-Prove that the sum of two even integers is even.}" \
  --top_k "${TOP_K:-3}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-1024}"
