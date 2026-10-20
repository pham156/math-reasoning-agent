# GRPO Seed Confirmation And KL/Clipping Controls

This is a small development-only study, not a new benchmark sweep or a Cartesian hyperparameter grid. Every new run starts from the untouched Qwen2.5-Math-1.5B base checkpoint and uses the same frozen512 MATH Level3-5 training problems,300 disjoint development problems,128 updates, four responses per prompt, temperature1, constant learning rate, FP32 parameters/BF16 compute, and binary exact/Math-Verify final-answer reward. Evaluation preserves the common3000-token cap and endoftext/im_end stopping. No retrieval, trained verifier, SFT, or new reward shaping is added.

## Matrix

| Condition | Seed | LR | KL loss coefficient | Clip distances (lower, upper) | Status |
|---|---:|---:|---:|---|---|
| Original control | 0 | 1e-6 | 0 | 0.2,0.2 | Reuse completed pilot/dev300 |
| Original higher LR | 0 | 3e-6 | 0 | 0.2,0.2 | Reuse completed upward sweep/dev300 |
| Seed confirmation control | 1 | 1e-6 | 0 | 0.2,0.2 | New |
| Seed confirmation higher LR | 1 | 3e-6 | 0 | 0.2,0.2 | New |
| Weak reference penalty | 0 | 1e-6 | 0.001 | 0.2,0.2 | New |
| Stronger reference penalty | 0 | 1e-6 | 0.01 | 0.2,0.2 | New |
| Narrow clipping | 0 | 1e-6 | 0 | 0.1,0.1 | New |
| Wide clipping | 0 | 1e-6 | 0 | 0.3,0.3 | New |

KL/clipping are anchored to the original seed0/1e-6 control, not combined with the best observed LR. Compare learning rates within each seed and report both paired deltas plus two-seed mean/sample standard deviation. Two seeds do not establish significance or an optimum. Temperature, rollout count, schedule, and batch-size sweeps are deferred.

## Controls And Diagnostics

`src.training.rl.pilot` exposes `--kl_beta`, `--clip_lower`, and `--clip_upper`. Defaults preserve previous runs; clipping overrides and nonzero KL are restricted to GRPO. Resume rejects changes in LR, seed, penalty, clipping, data, or budgets. The shared method profiles remain strict for other algorithms. Actual pinned TRL0.25.1 loss/gradient tests check that the KL coefficient and clipping bounds change the intended mathematical loss and gradients.

The KL coefficient multiplies TRL's reference-policy KL loss; it does not replace or reshape the binary correctness reward. The reference is reloaded from the original local base checkpoint on each restart, uses FP32 parameters, and is explicitly frozen. BF16 compute is preserved. Reference initialization kwargs specify local-only/FP32/SDPA; TRL may warn these kwargs are ignored for the already-loaded policy, while its separate reference loader uses them. Runtime precision/freeze assertions guard this behavior. See the pinned [TRL0.25.1 documentation](https://huggingface.co/docs/trl/v0.25.1/grpo_trainer) and locally inspected trainer source.

Each KL/clipping condition has its own eight-update continuous-versus-restarted GPU gate before its pilot. The gate compares checkpoint4, sampled rollouts, final parameters (tolerance1e-7), and objective settings. KL gates additionally require a frozen reference and finite positive measured training KL; clipping gates need not activate clipping naturally in eight updates, since synthetic loss/gradient tests validate bound sensitivity. Never bypass a failed gate. Independent experiments are not gated behind each other. Seed-only runs reuse the passed original backend restart gate.

Per-update histories retain reward and group reward variance, mixed-group fraction, entropy, gradient norms, response length, truncation, and actual clipping fractions. KL runs also record the trainer's sampled on-policy reference-KL estimator. Zero-penalty runs have no training KL measurement; missing values are reported as N/A, not zero. Loss, gradient norms, KL, entropy, and clip metrics fail on nonfinite values.

For a common drift comparison, `src.evaluation.grpo_drift` freezes eight development contexts from existing baseline responses. Each uses at most256 response tokens and16 evenly spaced next-token positions. It computes full-vocabulary KL(policy||base) exactly at those fixed prefixes with FP32 weights/BF16 compute. This is a deterministic conditional-distribution drift diagnostic, not sequence-level KL or an on-policy estimate; it is deliberately distinct from the training KL statistic. The same token IDs/vocabulary hash/probe hash apply to all eight models, including both reused controls. These probes do not feed gradients or rewards.

## Execution

Use a fresh scratch root, for example `$OUTPUT_DIR/rl_1p5b_study/grpo_controls_20261017`. Prepare probes on CPU:

```bash
source scripts/setup_env.sh
ROOT="$OUTPUT_DIR/rl_1p5b_study/grpo_controls_20261017"
bash scripts/rl/smoke_python.sh -m src.evaluation.grpo_drift --prepare \
  --probe_file "$ROOT/probes.json" \
  --controls "$OUTPUT_DIR/rl_1p5b_study/controls_exploration_20261014"
/usr/bin/python3 -m scripts.rl.submit_grpo_ablations --root "$ROOT"
/usr/bin/python3 -m scripts.rl.submit_grpo_ablations --root "$ROOT" --submit
```

The host stdlib submitter validates reused configs/models and performs Slurm dry runs before submitting. Runtime remains the isolated scratch Apptainer Python3.11 stack; no FuseMoE dependency changes. Jobs use ai-forge/a100-80gb/standby, GPU1,64G RAM, at most4hours, and job-ID-first home logs. All large artifacts/caches/checkpoints stay scratch. No artificial task throttle is added.

There are49 Slurm tasks: four independent restart gates,24 training continuation segments (four per new run),18 development-evaluation tasks (three per new run), two reused-control drift checks, and one combined summary. New drift scoring runs at the end of each model's evaluation shard0. Completed training continuations are no-ops. Summary uses genuine afterok dependencies and rejects incomplete128-update/300-row results or mismatched data, prompts, settings, or drift probes. Safe evaluation pauses may require rerunning an incomplete shard before summary succeeds.

Artifacts: `submission.json` records the exact conditions and job dependencies; `runs/<label>/train_history.json` stores curves; `evaluations/<label>/greedy/grpo/` stores per-problem JSONL; `diagnostics/<label>.json` stores fixed-prefix KL. `summary.{md,json}` includes paired-seed LR results, ablation accuracy/fixes/breaks, truncation, tokens/runtime, reward/group statistics, entropy, training KL, clipping activity, and common drift. Training job IDs support total Slurm GPU-time accounting; last-segment runtime is not the full training cost. Do not use MATH-500/GSM8K/AIME for tuning this grid.
