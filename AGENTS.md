# Project Guidelines

## Software Engineering & Code Architecture
- Write clean, readable, maintainable, and idiomatic Python following SOLID, KISS, DRY, and separation of concerns.
- Decouple core algorithms, configuration, I/O, Slurm orchestration, and evaluation into cohesive modules.
- Prefer composition over inheritance, explicit dependencies over hidden state, and structural Protocols over deep class hierarchies.
- Use declarative, typed, validated configurations; avoid hardcoded paths or hyperparameter magic numbers.
- Apply Test-Driven Development (TDD): write minimal failing tests for new functionality and regression tests for bug fixes before implementation.
- Keep changes focused and atomic; avoid unrelated refactoring or modifying historical baseline code.

## Research Rigor & Mathematical Reproducibility
- Preserve frozen dataset splits, random seeds (e.g. split seed 42), mathematical objectives, reward functions, and evaluation protocols unless explicitly authorized.
- Require `full_determinism`, deterministic PyTorch operations, cuBLAS workspace config, disabled cuDNN benchmarking, and disabled TF32 before model initialization.
- Maintain full FP32 parameters / BF16 compute across all training and evaluation passes.
- Distinguish completed empirical results from working hypotheses and proposed experiments. Follow `docs/rl_experiment_plan.md`.
- Evaluate using the common-stop protocol (tokens 151643/151645) and dual-report historical lenient reward vs strict `closed-boxed-v1` accuracy.
- `src.training.rl.sft_pilot` is the matched-data, full-parameter SFT control; ground-truth solutions remain strictly confined to the 512 frozen training IDs. Run CPU `--prepare_only` audits before GPU submission.

## Cluster Runtime & Execution (Purdue Gilbreth)
- Source `scripts/setup_env.sh` for canonical storage paths.
- Storage quarantine: Code, small job-ID-first Slurm logs, and `figures/` live in home; models, datasets, checkpoints, JSONL outputs, venvs, and package caches live strictly under `SCRATCH_ROOT`. Never write large artifacts to home.
- Execute via the isolated Apptainer runtime using `scripts/rl/smoke_python.sh`. Do not install packages into host environments, FuseMoE, or historical venvs.
- Slurm job discipline:
  - Submissions to `a100-80gb` must specify `--qos=standby`, `--time <= 03:59:00`, and explicit `--gres=gpu:1`.
  - Always run `bash -n` syntax checks and `sbatch --test-only` dry runs before submitting jobs. Do not submit GPU jobs without authorization.
  - Chain checkpoint dependencies within experiments (`afterok`) without artificial barriers between independent models.
- Restart & Resume gate protocol:
  - Never bypass a failed restart check or silently resume incomplete rollout batches. Continue only at 4-update rollout boundaries with `--save_steps 4`.
  - Verify `checkpoint-4` equality against the 1e-7 weight difference threshold to distinguish numerical divergence from resume failure.
  - Training segments must stop at safe boundaries after 2.5 hours to fit within 4-hour Slurm limits.
- Lean 4 formal proof verification:
  - Pinned Lean/Mathlib 4.24.0 via `scripts/lean/lean.sh`. Preflight packs the runtime into a read-only SquashFS image to eliminate Lustre metadata latency.
  - Generate tactic proofs inside the Python GPU container, then invoke the host standard-library proof driver to launch the separate Lean container (nested Apptainer is unsupported).
  - Enforce conservative proof policies (reject `sorry`, enforce 4 GiB memory / 180s timeout, audit allowed axioms via `#print axioms`).

## Agent Workflow
1. **Inspect**: Examine existing code, conventions, and cluster configurations before modifying files.
2. **Design**: Choose the simplest maintainable architecture with clear module boundaries.
3. **Test-First**: Write unit tests or regression checks in `tests/` before writing implementation code.
4. **Verify**: Execute verification checks inside `smoke_python.sh` and review the git diff.
5. **Report**: Report exactly what changed, what passed, and what remains unverified. Evidence before assertions always.

Keep documentation paragraphs on single lines.
