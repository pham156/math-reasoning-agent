#!/usr/bin/env bash

set -euo pipefail

cd /home/pham156/math-reasoning-agent

submit() {
  local label=$1
  shift
  local job_id
  job_id=$(sbatch --parsable "$@")
  job_id=${job_id%%;*}
  printf '%-28s %s\n' "$label" "$job_id" >&2
  printf '%s' "$job_id"
}

echo "Submitting QLoRA experiments with standby QoS." >&2

TRAIN_JOB=$(submit "training" scripts/run_finetune.slurm)

BASE_M500=$(submit "baseline MATH-500" scripts/run_det_math500_no_retrieval.slurm)
BASE_A24=$(submit "baseline AIME 2024" scripts/run_det_aime2024_baseline.slurm)
BASE_A25=$(submit "baseline AIME 2025" scripts/run_det_aime2025_baseline.slurm)

QLORA_M500=$(submit "QLoRA MATH-500" --dependency="afterok:$TRAIN_JOB" scripts/run_qlora_math500.slurm)
QLORA_A24=$(submit "QLoRA AIME 2024" --dependency="afterok:$TRAIN_JOB" scripts/run_qlora_aime2024.slurm)
QLORA_A25=$(submit "QLoRA AIME 2025" --dependency="afterok:$TRAIN_JOB" scripts/run_qlora_aime2025.slurm)
QLORA_M500_B5=$(submit "QLoRA MATH-500 best-of-5" --dependency="afterok:$TRAIN_JOB" scripts/run_qlora_math500_best_of_5_array.slurm)
QLORA_A24_B5=$(submit "QLoRA AIME 2024 best-of-5" --dependency="afterok:$TRAIN_JOB" scripts/run_qlora_aime2024_best_of_5.slurm)
QLORA_A25_B5=$(submit "QLoRA AIME 2025 best-of-5" --dependency="afterok:$TRAIN_JOB" scripts/run_qlora_aime2025_best_of_5.slurm)

DEPENDENCIES=$(IFS=:; echo "${BASE_M500}:${BASE_A24}:${BASE_A25}:${QLORA_M500}:${QLORA_A24}:${QLORA_A25}:${QLORA_M500_B5}:${QLORA_A24_B5}:${QLORA_A25_B5}")
COMPARE_JOB=$(submit "merge and comparison" --dependency="afterok:$DEPENDENCIES" scripts/run_qlora_merge_compare.slurm)

cat >&2 <<EOF

Training job: $TRAIN_JOB
Final comparison job: $COMPARE_JOB
Adapter: /scratch/gilbreth/pham156/math-reasoning-agent/outputs/qwen2.5-math-7b-math-lora
Summary: /scratch/gilbreth/pham156/math-reasoning-agent/outputs/qlora_eval/compare_qlora.md
EOF
