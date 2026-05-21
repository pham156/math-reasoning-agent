#!/bin/bash
#SBATCH --job-name=math-agent-qlora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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

python src/finetune_qlora.py \
  --output_dir "$OUTPUT_DIR/qwen2.5-math-7b-math-lora" \
  --max_seq_length "${MAX_SEQ_LENGTH:-2048}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-2e-4}" \
  --num_train_epochs "${EPOCHS:-1}"
