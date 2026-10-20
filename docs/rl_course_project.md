# An Empirical Study of Policy-Gradient Methods for RLVR in Mathematical Reasoning

The current proposed protocol is [the 1.5B experiment plan](rl_experiment_plan.md): 5,000 MATH training problems, no-RL controls, GRPO/Dr. GRPO/DAPO-style/GSPO as peer methods, and a fixed-optimizer reward study comparing correctness-only, format bonuses, and length penalties. It evaluates behavior, optimization, and compute as well as accuracy; partial/process rewards require validated graders. REINFORCE/PPO and broader alignment methods are optional supporting comparisons. The 7B setup and matrix below are the historical pilot roadmap, not the new launch configuration.

## Scope

The project is a controlled empirical study, not a new-algorithm claim. The primary model is Qwen2.5-Math-7B-Instruct and the primary reward is exact normalized final-answer correctness. Retrieval, verifier selection, consensus inference, and QLoRA remain supporting baselines.

The experimental sequence first measures whether the frozen policy can explore rewarding trajectories, then isolates training distribution and optimization effects, and treats response length as a diagnostic alongside accuracy and compute cost.

## Repository audit

The existing GRPO pilot is operational with TRL 1.10.0, 4-bit NF4 LoRA, four generations per prompt, batch size 1, gradient accumulation 4, 512 completion tokens, and 50 updates. The original learning rate was `1e-6`. The stabilization study found `2e-7` best on the fixed first 50 MATH-500 examples (86% versus the 84% frozen baseline), but this one-example gain is only a development signal.

Current logging captures reward and completion length. TRL can expose KL and entropy, but KL is absent when the GRPO KL coefficient is zero and the old CSV field list misses several modern TRL metric names. Correct/incorrect response lengths and mixed-group rates require rollout-level logging.

TRL 1.10.0 provides production `GRPOTrainer`. PPO is available only through `trl.experimental.ppo` and requires learned reward/value model objects, so it is not directly compatible with the callable binary reward. TRL has no REINFORCE trainer. PPO and REINFORCE therefore require dedicated memory-safe adapters before real runs; capability commands fail clearly until that work is complete.

## Staged protocol

1. Stage 1: validate reward/objective tests, materialize fixed subsets, and run frozen-policy exploration on 50 problems for temperatures 0.4/0.7/1.0 and rollout counts 2/4/8.
2. Stage 2: after selecting exploration settings, compare REINFORCE, PPO, and GRPO on the same 500-example subset, 50-update budget, seed, prompt, and first-50 MATH-500 gate.
3. Stage 3: use the stable method on equal-size MATH level 1-2, MATH level 3-5, random MATH, and GSM8K subsets.
4. Stage 4: one-factor-at-a-time scaling over data size, updates, and rollout count.
5. Stage 5: full MATH-500, GSM8K, AIME 2024, and AIME 2025 evaluation. Greedy accuracy and sampled pass@k remain separate metrics.

## Proposed Stage 2 matrix

| run | algorithm | train n | updates | rollouts | temperature | learning rate | seed | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| reinforce_common50 | REINFORCE | 500 | 50 | 4 | 0.7 | 2e-7 | 0 | blocked on trainer adapter |
| ppo_common50 | PPO | 500 | 50 | 4 | 0.7 | 2e-7 | 0 | blocked on callable-reward/value adapter |
| grpo_common50 | GRPO | 500 | 50 | 4 | 0.7 | 2e-7 | 0 | ready after exploration selection |

This matrix is intentionally not submitted yet. A fair comparison also needs a token-budget report because PPO performs multiple epochs and uses a value model while GRPO spends compute on grouped rollouts.

## Runtime estimate

On one A100 80GB, prior 50-step GRPO variants took roughly 57-61 minutes end to end; 25 steps took about 40 minutes. Training itself took about 33 minutes in the first pilot. A 50-problem frozen-policy exploration task is expected to vary approximately linearly with rollout count, and the nine-task grid is constrained to two concurrent jobs. GPU peak memory was not recorded by the old pilot, so group size 8 remains a measured feasibility condition, not an assumed-safe setting.

## Commands

```bash
source scripts/setup_env.sh
conda activate /scratch/gilbreth/pham156/conda_envs/math-reasoning-grpo
export PYTHONNOUSERSITE=1

python -m src.training.rl.experiment_matrix --capabilities
python -m src.training.rl.experiment_matrix
python -m src.training.sampling.math_subsets --subset math_level_3_5 --limit 1000 --seed 0

sbatch --test-only scripts/rl/analyze_exploration.slurm
# Submit only after reviewing the matrix:
# sbatch scripts/rl/analyze_exploration.slurm
```

## Method mechanics

REINFORCE multiplies the completion sequence log-probability by the observed return minus a batch baseline. PPO adds a learned value baseline/GAE and clips the policy probability ratio, while also constraining drift with KL. GRPO avoids a value model by centering and scaling rewards within each prompt's rollout group, then applies a clipped policy update. The shared objective tests use token-summed losses so response length is not silently treated as an optimization target.
