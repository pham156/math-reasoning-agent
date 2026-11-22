#!/usr/bin/env bash
#SBATCH --job-name=aime_eval_a10
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}}"
cd "$PROJECT_ROOT"
source scripts/setup_env.sh >/dev/null

export RL_SMOKE_GPU=1

RL_AIME_ROOT="${RL_AIME_ROOT:-$OUTPUT_DIR/rl_1p5b_study/aime_20261122}"
RL_PILOT_ROOT="${RL_PILOT_ROOT:-$OUTPUT_DIR/rl_1p5b_study/pilot_20261122_004558}"
RL_COMPARISON_ROOT="${RL_COMPARISON_ROOT:-$OUTPUT_DIR/rl_1p5b_study}"
SFT_MODEL_PATH="$OUTPUT_DIR/rl_1p5b_study/sft_pilot/final_model"

echo "=== Running AIME 2025/2026 Evaluation on A10 GPU ==="
echo "Node: $SLURMD_NODENAME | GPU: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

# 1. Baseline greedy and sampled (16 samples) on AIME 2025 and 2026
for year in 2025 2026; do
  echo "--- Evaluating Baseline on AIME $year (greedy) ---"
  bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_exploration \
    --pilot_root "$RL_PILOT_ROOT" \
    --comparison_root "$RL_COMPARISON_ROOT" \
    --name baseline \
    --shard 0 \
    --benchmark_file "$RL_AIME_ROOT/aime${year}_frozen.json" \
    --output_dir "$RL_AIME_ROOT/aime${year}" \
    --mode greedy

  echo "--- Evaluating Baseline on AIME $year (sampled 16) ---"
  bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_exploration \
    --pilot_root "$RL_PILOT_ROOT" \
    --comparison_root "$RL_COMPARISON_ROOT" \
    --name baseline \
    --shard 0 \
    --benchmark_file "$RL_AIME_ROOT/aime${year}_frozen.json" \
    --output_dir "$RL_AIME_ROOT/aime${year}" \
    --mode sampled \
    --num_samples 16
done

# 2. SFT Pilot greedy on AIME 2025 and 2026
for year in 2025 2026; do
  echo "--- Evaluating SFT on AIME $year (greedy) ---"
  bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_exploration \
    --pilot_root "$RL_PILOT_ROOT" \
    --comparison_root "$RL_COMPARISON_ROOT" \
    --model_path "$SFT_MODEL_PATH" \
    --name sft \
    --shard 0 \
    --benchmark_file "$RL_AIME_ROOT/aime${year}_frozen.json" \
    --output_dir "$RL_AIME_ROOT/aime${year}" \
    --mode greedy
done

# 3. Summarize results
echo "=== Generating AIME summaries ==="
bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_benchmarks "$RL_AIME_ROOT" --suite aime --models baseline sft --modes greedy
cp "$RL_AIME_ROOT/aime_summary.md" "$RL_AIME_ROOT/aime_greedy_summary.md"
cp "$RL_AIME_ROOT/aime_summary.json" "$RL_AIME_ROOT/aime_greedy_summary.json"

bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_benchmarks "$RL_AIME_ROOT" --suite aime --models baseline --modes sampled
cp "$RL_AIME_ROOT/aime_summary.md" "$RL_AIME_ROOT/aime_sampled_summary.md"
cp "$RL_AIME_ROOT/aime_summary.json" "$RL_AIME_ROOT/aime_sampled_summary.json"

echo "=== Evaluation complete ==="
