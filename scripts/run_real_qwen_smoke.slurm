#!/usr/bin/env bash
#SBATCH --job-name=math_qwen_smoke
#SBATCH --account=ai-forge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pham156/math-reasoning-agent/logs/%x_%j.out
#SBATCH --error=/home/pham156/math-reasoning-agent/logs/%x_%j.err

set -euo pipefail

cd /home/pham156/math-reasoning-agent

module load conda/2024.09
conda activate /home/pham156/math-reasoning-agent/env/math-reasoning-agent

# shellcheck source=/dev/null
source scripts/setup_env.sh

MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/Qwen2.5-Math-7B-Instruct}"
PROBLEM="${PROBLEM:-Find all real x such that x^2 - 5x + 6 = 0.}"
TOP_K="${TOP_K:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/smoke_real_qwen.json}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "SCRATCH_ROOT=$SCRATCH_ROOT"
echo "MODEL_PATH=$MODEL_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"

python -m src.agents.pipeline \
  --problem "$PROBLEM" \
  --top_k "$TOP_K" \
  --retrieval_source math \
  --model_path "$MODEL_PATH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --output "$OUTPUT_PATH"
