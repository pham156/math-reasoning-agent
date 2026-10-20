# math-reasoning-agent

An AI-heavy mathematical reasoning agent for competition and proof-style math on Purdue Gilbreth.

Git excludes generated results, job logs/state, checkpoints, local environments, runtime images, and compiled paper/figure exports. Source code, experiment configs, Slurm scripts, LaTeX, and container recipes remain versioned; ignored local artifacts are not deleted.

The optional `.vendor-CS557/` checkout and `open-r1` convenience symlink are local references, not vendored project dependencies, and are excluded from Git. The project's implemented training and evaluation modules live under `src/`; cloning this repository does not require that external checkout. Local `.env` files are also ignored; never commit credentials.

The [two-page IEEE midterm update](docs/midterm/README.md) includes a compiled PDF, line plots, and an Overleaf-ready ZIP. The October 19, 2026 revision focuses on GRPO and Dr. GRPO, full-parameter fine-tuning, an initial SFT control, exploration, external benchmarks, paired-seed learning rates, and KL/clipping controls.

The current research direction is **An Empirical Study of Policy-Gradient Methods for Reinforcement Learning with Verifiable Rewards in Mathematical Reasoning**. See [the RL course project roadmap](docs/rl_course_project.md). Retrieval, verification, consensus, and QLoRA remain supporting baselines.

The [compact 1.5B experiment plan](docs/rl_experiment_plan.md) specifies 5,000 MATH training problems, no-RL controls, a comparison of GRPO, Dr. GRPO, DAPO-style optimization, and GSPO, and a separate correctness/format/length reward-design study with a fixed optimizer. It also defines full-parameter BF16 feasibility checks, behavior and optimization diagnostics, and checkpointed Gilbreth execution. Partial/process rewards require validated graders; dynamic sampling, REINFORCE++ with a group baseline, and verified-solution rejection fine-tuning are additional comparisons.

## Four-Algorithm GPU Smoke Test

The [GRPO seed-confirmation and KL/clipping study](docs/grpo_control_study.md) defines six new one-factor pilots, two reused controls, per-condition restart gates, paired-seed evaluation, and distinct training-KL/fixed-prefix-drift diagnostics. Temperature, rollout count, schedule, and batch-size sweeps remain deferred.

For a GRPO-only upward learning-rate pilot, use `/usr/bin/python3 scripts/rl/submit_lr_pilots.py --root "$OUTPUT_DIR/rl_1p5b_study/grpo_lr_upper_20261016" --algorithms grpo --learning_rates 2e-6 3e-6` for the Slurm dry run, then add `--submit` to launch. The completed1e-6 control is automatically reused. This creates two fresh base-model runs at fixed512 training problems/128updates/seed0, with all other settings unchanged: eight checkpoint training segments, six dev300 evaluation tasks, and one summary task (15tasks via11sbatch submissions). Independent rates have no cross-run dependency or throttle; only each run's checkpoints and evaluation are chained. Outputs/caches remain scratch and job-ID-first logs stay home. These are development-only tuning runs, not MATH-500/GSM8K model selection; no additional hyperparameter dimensions are crossed.

`scripts/rl/run_algorithm_smokes.slurm` runs GRPO, Dr. GRPO, DAPO-style optimization, and GSPO as a four-task array with at most two concurrent A100 80GB jobs on `ai-forge/standby`. Each task uses 64 Level 3-5 MATH train problems, four rollouts, five updates, a 3,000-token cap, and two disjoint train-split problems for before/after inference. These are runtime checks, not benchmark results. Full policy weights and optimizer state are FP32, with BF16 compute; no LoRA or quantization is used.

The isolated Python 3.11 Apptainer runtime uses `$SCRATCH_ROOT/env/rl-smoke`, separate from both FuseMoE and the historical GRPO environment. Models, dependencies, all caches, temporary files, checkpoints, and results stay in scratch. Only code and job-ID-first Slurm logs live in home. Objective loss/gradient tests validate the pinned TRL 0.25.1 backend, including a DAPO generation-batch normalization correction. An exact interrupted-resume test is still required before longer chained training.

```bash
source scripts/setup_env.sh
bash scripts/rl/setup_smoke_runtime.sh
bash scripts/rl/smoke_python.sh -m unittest tests.test_rl_smoke tests.test_rlvr_rewards
sbatch --test-only scripts/rl/run_algorithm_smokes.slurm
sbatch scripts/rl/run_algorithm_smokes.slurm
# Replace ARRAY_JOB_ID with the submitted ID:
sbatch --dependency=afterany:ARRAY_JOB_ID --export=ALL,SMOKE_ARRAY_JOB_ID=ARRAY_JOB_ID scripts/rl/summarize_algorithm_smokes.slurm
```

Artifacts are saved under `$OUTPUT_DIR/rl_1p5b_study/smoke/ARRAY_JOB_ID/`: one folder per method, plus `summary.json` and `summary.md`. Each method records objective settings, versions, dataset IDs, raw rollouts and correctness rewards, before/after solutions, training history, peak memory, parameter-change checks, and checkpoint paths. Missing or failed tasks are included in the summary.

The first version combines:

- Qwen and DeepSeek math model inference
- MATH dataset loading
- FAISS retrieval over solved examples
- retrieval-augmented solution generation
- verifier and iterative self-critique revision loop
- conservative QLoRA fine-tuning scaffold
- future FastAPI/React visualization path

## Storage Rule

Large files must stay out of `/home`.

Code, scripts, notebooks, environments, and logs live in home:

```bash
/home/pham156/math-reasoning-agent
```

Models, datasets, checkpoints, caches, temporary files, and bulky outputs live in scratch:

```bash
/scratch/gilbreth/pham156/math-reasoning-agent
```

The setup script keeps those two locations separate.

## Directory Layout

```text
/home/pham156/math-reasoning-agent/
  logs/
  src/
  scripts/
  notebooks/
  env/

/scratch/gilbreth/pham156/math-reasoning-agent/
  models/
  data/
  outputs/
  cache/
```

## Source Layout

Implementation code is organized by subsystem:

```text
src/core/        shared config and baseline model inference
src/retrieval/   MATH indexing, FAISS retrieval, reranking, prompt context
src/agents/      solver agent, self-critique loop, end-to-end pipeline
src/verifier/    verifier prompts, schemas, scoring, backend adapters
src/proof/       proof-search tree, branch scoring, visualization
src/evaluation/  answer matching, benchmark runner, aggregate metrics
src/training/    QLoRA fine-tuning entrypoint
```

The canonical entrypoints use package modules such as
`python -m src.agents.run_pipeline` and `python -m src.retrieval.build_index`.
Older direct-script wrappers were moved to `src/legacy_wrappers/`; do not use
them for new commands.

## Environment Setup

From the project root:

```bash
cd /home/pham156/math-reasoning-agent
source scripts/setup_env.sh
```

Create a conda environment if conda is available:

```bash
conda env create -f environment.yml
conda activate math-reasoning-agent
source scripts/setup_env.sh
```

Or use venv:

```bash
python -m venv env/math-reasoning-agent
source env/math-reasoning-agent/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
source scripts/setup_env.sh
```

The setup script exports:

```bash
PROJECT_ROOT=/home/pham156/math-reasoning-agent
SCRATCH_ROOT=/scratch/gilbreth/pham156/math-reasoning-agent
MODEL_DIR=$SCRATCH_ROOT/models
DATA_DIR=$SCRATCH_ROOT/data
OUTPUT_DIR=$SCRATCH_ROOT/outputs
CACHE_DIR=$SCRATCH_ROOT/cache
LOG_DIR=$PROJECT_ROOT/logs
HF_HOME=$CACHE_DIR/huggingface
TRANSFORMERS_CACHE=$CACHE_DIR/huggingface/transformers
HF_DATASETS_CACHE=$CACHE_DIR/huggingface/datasets
TORCH_HOME=$CACHE_DIR/torch
WANDB_DIR=$LOG_DIR/wandb
TMPDIR=$CACHE_DIR/tmp
```

## Download Data

```bash
source scripts/setup_env.sh
python scripts/download_data.py
```

This downloads `hendrycks/competition_math` and saves it to:

```bash
$DATA_DIR/MATH
```

## Download Models

```bash
source scripts/setup_env.sh
bash scripts/download_models.sh
```

Models are saved under `$MODEL_DIR/`:

- `Qwen2.5-Math-7B-Instruct`
- `Qwen2.5-Math-7B`
- `deepseek-math-7b-instruct`
- `all-MiniLM-L6-v2`

The script prints the disk locations before downloading and refuses non-scratch project roots.

## Build Retrieval Index

```bash
source scripts/setup_env.sh
python -m src.retrieval.build_index
```

Equivalent package entrypoint:

```bash
python -m src.retrieval.build_index
```

Outputs:

```bash
$DATA_DIR/faiss/math.index
$DATA_DIR/faiss/metadata.jsonl
$DATA_DIR/faiss/index_config.json
```

Test retrieval:

```bash
python -m src.retrieval.retrieve --problem "Find all integer solutions to x^2 - y^2 = 15." --top_k 3
```

Test the package entrypoint after the index exists:

```bash
python -m src.retrieval.retrieve \
  --problem "Find all real x such that x^2 - 5x + 6 = 0." \
  --top_k 3
```

The retrieval system is intentionally more than semantic nearest-neighbor search:

- `src/retrieval/math_taxonomy.py` infers topic/subfield, proof strategy, and theorem-pattern labels.
- `src/retrieval/embedding.py` builds embedding text from problem, solution, topic, strategy, and theorem signals.
- `src/retrieval/retrieve.py` runs FAISS broad retrieval.
- `src/retrieval/rerank.py` reranks candidates using semantic score, topic overlap, strategy overlap, theorem-pattern overlap, lexical overlap, and solution-shape features.
- `src/retrieval/strategy_bank.py` stores compact theorem/proof-strategy descriptions for prompt hints.
- `src/retrieval/context.py` formats the final context for the solver LLM.

Useful retrieval inspection commands:

```bash
python -m src.retrieval.retrieve \
  --problem "Find all integer solutions to x^2 - y^2 = 15." \
  --top_k 4 \
  --candidate_k 40

python -m src.retrieval.retrieve \
  --problem "Prove that there are infinitely many primes." \
  --show_context

python -m src.retrieval.retrieve \
  --problem "Prove that there are infinitely many primes." \
  --show_solver_prompt
```

Evaluate retrieval quality:

```bash
python -m src.retrieval.evaluate_retrieval --split test --num_queries 200 --top_k 10
```

This writes:

```bash
$LOG_DIR/retrieval_eval_summary.json
$LOG_DIR/retrieval_eval_rows.csv
```

## Baseline Inference

```bash
python -m src.core.infer_baseline \
  --problem "Prove that the sum of two even integers is even." \
  --max_new_tokens 1024 \
  --temperature 0.2
```

To compare DeepSeek:

```bash
python -m src.core.infer_baseline \
  --model_path "$MODEL_DIR/deepseek-math-7b-instruct" \
  --problem "Solve for all real x such that x^2 - 5x + 6 = 0."
```

## End-to-End Agent Pipeline

```bash
python -m src.agents.run_pipeline --problem "Prove that there are infinitely many primes."
```

Pipeline:

1. retrieve similar solved examples from FAISS
2. build a RAG prompt
3. generate a candidate solution
4. verify for algebra mistakes, logical gaps, invalid assumptions, and missing justification
5. revise if the verifier marks the solution invalid or uncertain
6. print the final solution

## First End-to-End Smoke Test

This is the frozen minimal vertical slice. It works from the project root and
supports a deterministic mock LLM for testing without model weights. If the
FAISS/MATH index is unavailable in `--retrieval_source auto`, it uses a tiny toy
retrieval set.

```bash
cd /home/pham156/math-reasoning-agent
source scripts/setup_env.sh

python -m src.agents.pipeline \
  --problem "Find all real x such that x^2 - 5x + 6 = 0." \
  --top_k 3 \
  --max_new_tokens 512 \
  --mock_llm \
  --output outputs/smoke_test.json
```

Relative paths under `outputs/` are saved under scratch:

```bash
$OUTPUT_DIR/smoke_test.json
```

Run real Qwen2.5-Math-7B-Instruct inference after downloading the model to
scratch:

```bash
python -m src.agents.pipeline \
  --problem "Find all real x such that x^2 - 5x + 6 = 0." \
  --top_k 3 \
  --retrieval_source math \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --max_new_tokens 512 \
  --output outputs/smoke_real_qwen.json
```

The saved JSON includes the input problem, retrieved examples, solver prompt,
raw solution, verifier critique, final solution, runtime metadata, and model
metadata. Real mode exits with a clear error if model files are missing instead
of silently falling back to mock mode. The `$PROJECT_ROOT/models/...` path is
accepted for convenience; the pipeline resolves it to `$MODEL_DIR/...` under
scratch before loading model weights.

Submit the real-Qwen smoke test on Gilbreth:

```bash
sbatch scripts/run_real_qwen_smoke.slurm
```

Optional overrides:

```bash
MODEL_PATH="$MODEL_DIR/Qwen2.5-Math-7B-Instruct" \
OUTPUT_PATH="outputs/smoke_real_qwen.json" \
sbatch scripts/run_real_qwen_smoke.slurm
```

Use real MATH retrieval with the mock LLM:

```bash
python -m src.agents.pipeline \
  --problem "Find all real x such that x^2 - 5x + 6 = 0." \
  --top_k 3 \
  --mock_llm \
  --retrieval_source math \
  --output outputs/smoke_math_retrieval.json
```

If the MATH dataset or FAISS index is missing, run:

```bash
source scripts/setup_env.sh
python scripts/download_data.py
python -m src.retrieval.build_index
```

The real-retrieval output JSON includes `runtime_metadata.retrieval_metadata`
with the dataset path, FAISS index path, metadata path, whether the index was
built during the run, and the number of retrieved examples.

Control the self-critique loop:

```bash
python -m src.agents.run_pipeline \
  --problem "Prove that sqrt(2) is irrational." \
  --max_iterations 2 \
  --expected_answer "sqrt(2) is irrational"
```

## Verifier And Self-Critique

The verifier is modular so the prompt-based checker can later be replaced by a trained verifier model.

Core modules:

- `src/verifier/prompts.py`: verifier, critique, and revision prompt templates
- `src/verifier/schema.py`: structured `VerificationResult` and `VerificationIssue` schema
- `src/verifier/scoring.py`: fallback scoring from issue categories and severities
- `src/verifier/backends.py`: `VerifierBackend` interface and prompt-based backend
- `src/verifier/critique_generation.py`: deterministic or LLM-generated critique text
- `src/agents/self_critique_loop.py`: solver -> verifier -> revised solver loop

Verifier checks:

- algebra mistakes
- logical gaps
- unjustified claims
- incomplete proofs
- invalid assumptions
- final-answer mismatch when an expected answer is provided

Inspect the verifier prompt:

```bash
python -m src.verifier.agent \
  --problem "Solve x^2 - 5x + 6 = 0." \
  --solution "x=2" \
  --print_prompt
```

Run structured verification:

```bash
python -m src.verifier.agent \
  --problem "Solve x^2 - 5x + 6 = 0." \
  --solution "x=2"
```

Run the standalone self-critique loop:

```bash
python -m src.agents.self_critique_loop \
  --problem "Prove that the sum of two even integers is even." \
  --max_iterations 2
```

JSON trace output:

```bash
python -m src.agents.self_critique_loop \
  --problem "Prove that there are infinitely many primes." \
  --json
```

## Proof Search

The proof-search layer is a lightweight inference-oriented alternative to single-chain generation.
It generates multiple strategy roots, branches into candidate reasoning paths, scores branches, keeps the best branches, and prunes weak ones.

Supported strategy families include:

- induction
- contradiction
- invariant or monovariant
- inequalities
- constructive proof
- combinatorial counting

Core modules:

- `src/proof/tree.py`: proof tree node structure and traversal helpers
- `src/proof/prompts.py`: strategy generation, branch expansion, and proof completion prompts
- `src/proof/scoring.py`: heuristic branch scoring plus optional verifier scoring
- `src/proof/search.py`: beam-style search controller
- `src/proof/visualization.py`: text, JSON, and Mermaid export utilities

Run a lightweight proof search:

```bash
python -m src.proof.search \
  --problem "Prove that the sum of two even integers is even." \
  --max_depth 2 \
  --beam_width 3 \
  --branch_factor 2
```

Use retrieval context and verifier scoring:

```bash
python -m src.proof.search \
  --problem "Prove that sqrt(2) is irrational." \
  --use_retrieval \
  --use_verifier \
  --max_depth 2 \
  --beam_width 3
```

Export the tree as JSON:

```bash
python -m src.proof.search \
  --problem "Prove that there are infinitely many primes." \
  --format json \
  --output proof_search_primes.json
```

Export Mermaid for quick visualization:

```bash
python -m src.proof.search \
  --problem "Prove that there are infinitely many primes." \
  --format mermaid \
  --output proof_search_primes.mmd
```

Outputs with relative `--output` paths are written under `$LOG_DIR`.

## QLoRA Fine-Tuning

The script fine-tunes `Qwen/Qwen2.5-Math-7B` using MATH examples formatted as:

```text
### Problem:
{problem}

### Solution:
{solution}
```

Run a short smoke test first:

```bash
python -m src.training.finetune_qlora --max_train_samples 32 --save_steps 20
```

Full conservative first pass:

```bash
python -m src.training.finetune_qlora
```

Adapter output:

```bash
$OUTPUT_DIR/qwen2.5-math-7b-math-lora
```

The adapter directory also contains durable training diagnostics:

```text
train_metrics.json
trainer_state.json
log_history.jsonl
loss_curve.csv
loss_curve.png
```

Slurm stdout/stderr are kept in `$LOG_DIR` with the job ID first. Checkpoints,
the final adapter, metrics, and loss curves are written to `$OUTPUT_DIR` in
scratch.

Defaults:

- max sequence length: 2048
- batch size: 1
- gradient accumulation: 8
- learning rate: 2e-4
- epochs: 1
- 4-bit NF4 quantization
- bf16 when available
- LoRA target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Slurm

Baseline inference:

```bash
sbatch scripts/run_baseline.slurm
```

Override the problem:

```bash
sbatch --export=ALL,PROBLEM="Prove that sqrt(2) is irrational." scripts/run_baseline.slurm
```

QLoRA:

```bash
sbatch scripts/run_finetune.slurm
```

The training script uses `ai-forge`, `a100-80gb`, and `standby` with the
four-hour QoS limit. Resume an interrupted run from a scratch checkpoint with:

```bash
sbatch --export=ALL,RESUME_FROM_CHECKPOINT="$OUTPUT_DIR/qwen2.5-math-7b-math-lora/checkpoint-500" \
  scripts/run_finetune.slurm
```

QLoRA benchmark evaluation after the adapter is saved:

```bash
sbatch scripts/run_qlora_math500.slurm
sbatch scripts/run_qlora_aime2024.slurm
sbatch scripts/run_qlora_aime2025.slurm
sbatch scripts/run_qlora_math500_best_of_5_array.slurm
sbatch scripts/run_qlora_aime2024_best_of_5.slurm
sbatch scripts/run_qlora_aime2025_best_of_5.slurm
```

Merge the MATH-500 best-of-5 shards and compare against the deterministic
Qwen2.5-Math-7B-Instruct baselines:

```bash
sbatch scripts/run_qlora_merge_compare.slurm
```

To submit the complete dependency graph, including fresh Instruct baselines,
training, QLoRA deterministic runs, QLoRA best-of-5 runs, and final summaries:

```bash
bash scripts/submit_qlora_experiments.sh
```

The submission script prints every job ID. Evaluation jobs use `afterok` on
training, and the final comparison uses `afterok` on all benchmark jobs.

The QLoRA comparison writes:

```text
outputs/qlora_eval/compare_qlora.md
outputs/qlora_eval/compare_qlora.json
```

Run the aligned 50-example base-model control and QLoRA failure analysis with:

```bash
sbatch scripts/run_qlora_diagnostic_base50.slurm
```

This uses the same deterministic evaluation prompt as the existing QLoRA run
and writes:

```text
outputs/qlora_diagnostics/math500_base50.jsonl
outputs/qlora_diagnostics/qlora_failure_analysis.md
```

### Corrected Small QLoRA Validation

Before running another full QLoRA benchmark, run a prompt-matched small
validation. This experiment fine-tunes from `Qwen2.5-Math-7B-Instruct`, uses
the same chat-template user prompt as evaluation, trains on only 1000 MATH
train examples, and evaluates only the first 50 MATH-500 examples.

```bash
sbatch scripts/run_qlora_instruct_small50.slurm
```

Defaults:

- base model: `$MODEL_DIR/Qwen2.5-Math-7B-Instruct`
- adapter output: `$OUTPUT_DIR/qwen2.5-math-7b-instruct-math1000-lora-lr5e-5`
- prompt format: `chat_eval`
- max train examples: `1000`
- learning rate: `5e-5`
- epochs: `1`
- evaluation: first 50 MATH-500 examples only

The job writes:

```text
outputs/qlora_instruct_small/math500_first50_instruct_baseline.jsonl
outputs/qlora_instruct_small/math500_first50_instruct_lora.jsonl
outputs/qlora_instruct_small/small50_comparison.md
outputs/qlora_instruct_small/small50_comparison.json
```

The comparison exits with status `3` if the adapter is not strictly better than
the Instruct baseline on the first 50 examples. Treat that as the stop signal:
do not run AIME or full MATH-500 for that adapter.

### Small RLVR / GRPO Validation

The RLVR path uses a separate scratch conda environment because the original
project env does not expose TRL's `GRPOTrainer`/`GRPOConfig`.

```bash
module load conda/2024.09
conda activate /scratch/gilbreth/pham156/conda_envs/math-reasoning-grpo
source scripts/setup_env.sh
export PYTHONNOUSERSITE=1
```

Check local GRPO support without training:

```bash
env PYTHONNOUSERSITE=1 python -m src.training.train_grpo_rlvr \
  --output_dir "$OUTPUT_DIR/grpo_rlvr_capability_check" \
  --capability_only
```

The first RLVR experiment is:

```bash
sbatch scripts/run_grpo_rlvr_small50.slurm
```

Defaults:

- model: `$MODEL_DIR/Qwen2.5-Math-7B-Instruct`
- train examples: `500`
- generations per prompt: `4`
- max steps: `50`
- learning rate: `1e-6`
- LoRA policy updates
- reward: `1.0` when normalized extracted final answer matches the MATH
  ground-truth final answer, otherwise `0.0`
- evaluation: first 50 MATH-500 examples only

The job writes training artifacts under:

```text
$OUTPUT_DIR/grpo_rlvr_qwen2.5_math7b_instruct_math500/
```

and validation reports under:

```text
outputs/rlvr_grpo_small/math500_first50_instruct_baseline.jsonl
outputs/rlvr_grpo_small/math500_first50_rlvr.jsonl
outputs/rlvr_grpo_small/first50_report.md
outputs/rlvr_grpo_small/first50_report.json
```

The report exits with status `3` if the first-50 validation drops below the
Instruct baseline, reward collapses, generations look malformed, or response
length collapses. Do not run full MATH-500 or AIME unless this first-50 run
matches or beats the 84% Instruct baseline.

Adapter evaluation uses the base model plus adapter:

```bash
python -m src.evaluation.run_eval \
  --benchmark math500 \
  --model_path "$MODEL_DIR/Qwen2.5-Math-7B" \
  --adapter_path outputs/qwen2.5-math-7b-math-lora \
  --output outputs/qlora_eval/math500_qlora.jsonl
```

All Slurm logs are written to:

```bash
$LOG_DIR/
```

Log filenames put the Slurm job id first for easier tracking:

```text
$LOG_DIR/<job_id>_<job_name>.out
$LOG_DIR/<job_id>_<job_name>.err
```

## Evaluation Plan

Initial evaluation should stay simple:

1. Use MATH test split by category and level.
2. Generate one answer per problem with baseline Qwen.
3. Generate one answer per problem with RAG enabled.
4. Track exact answer extraction separately from reasoning quality.
5. Use the verifier score as a diagnostic, not as ground truth.
6. Add self-consistency later by sampling multiple candidates and majority voting on extracted final answers.

Useful metrics:

- exact final-answer accuracy after answer normalization
- verifier score distribution
- pass rate by MATH level
- retrieval ablation: no retrieval vs top-3 retrieval
- model comparison: Qwen instruct vs DeepSeek instruct vs fine-tuned adapter

## Evaluation Framework

The evaluation runner compares four modes on MATH subsets:

- `baseline`: single-chain LLM solver
- `retrieval`: retrieval-augmented solver
- `verifier`: retrieval solver plus solver-verifier revision loop
- `search`: proof-search solver with branch scoring and pruning

Metrics collected:

- exact answer accuracy using boxed/final-answer extraction
- verifier verdicts, scores, issue counts, and issue categories
- retrieval top-k, semantic score, rerank score, topics, and strategies
- proof-search branch counts, pruned/complete/failed counts, best branch score, and strategy counts
- runtime per example

Run a small benchmark:

```bash
python -m src.evaluation.evaluate_agent \
  --split test \
  --limit 20 \
  --modes baseline,retrieval,verifier,search \
  --experiment_name smoke_eval
```

Run only cheaper modes:

```bash
python -m src.evaluation.evaluate_agent \
  --split test \
  --limit 50 \
  --modes baseline,retrieval \
  --experiment_name baseline_vs_rag
```

Compare verifier statistics for every mode output:

```bash
python -m src.evaluation.evaluate_agent \
  --split test \
  --limit 20 \
  --modes baseline,retrieval,verifier,search \
  --verify_all_outputs \
  --experiment_name verified_comparison
```

Run search with retrieval and verifier scoring:

```bash
python -m src.evaluation.evaluate_agent \
  --split test \
  --limit 10 \
  --modes search \
  --search_use_retrieval \
  --search_use_verifier \
  --max_depth 2 \
  --beam_width 3 \
  --branch_factor 2 \
  --experiment_name search_eval
```

Outputs are written to:

```bash
$LOG_DIR/experiments/<experiment_name>/
  config.json
  run.log
  results.jsonl
  results.csv
  summary.json
```

Convenience scripts:

```bash
bash scripts/benchmark_math_subset.sh
sbatch scripts/run_eval.slurm
```

Override defaults:

```bash
LIMIT=10 MODES=baseline,retrieval EXPERIMENT_NAME=quick_eval bash scripts/benchmark_math_subset.sh
sbatch --export=ALL,LIMIT=10,MODES=baseline,retrieval,EXPERIMENT_NAME=quick_eval scripts/run_eval.slurm
```

### 20-Example Qwen Evaluation Slice

This narrow runner compares the current real-Qwen inference path without adding
fine-tuning or UI code.

No retrieval:

```bash
python -m src.evaluation.run_eval \
  --split test \
  --limit 20 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --retrieval_source none \
  --use_verifier false \
  --output outputs/eval_20_no_retrieval.jsonl
```

MATH retrieval:

```bash
python -m src.evaluation.run_eval \
  --split test \
  --limit 20 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --retrieval_source math \
  --use_verifier false \
  --output outputs/eval_20_retrieval.jsonl
```

MATH retrieval plus verifier:

```bash
python -m src.evaluation.run_eval \
  --split test \
  --limit 20 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --retrieval_source math \
  --use_verifier true \
  --output outputs/eval_20_retrieval_verifier.jsonl
```

Summarize any JSONL:

```bash
python -m src.evaluation.summarize outputs/eval_20_retrieval.jsonl
```

Slurm entrypoints:

```bash
sbatch scripts/run_eval_20_no_retrieval.slurm
sbatch scripts/run_eval_20_retrieval.slurm
sbatch scripts/run_eval_20_retrieval_verifier.slurm
```

### Benchmark Presets

The same runner supports benchmark-style presets commonly reported in reasoning
model papers. Each JSONL starts with a `run_metadata` row containing benchmark
metadata, sampling settings, model metadata, and retrieval settings.

Download benchmark datasets to scratch:

```bash
python scripts/download_benchmarks.py
```

This saves local copies under `$DATA_DIR` for MATH, MATH-500, GSM8K, AIME 2024,
AIME 2025, and miniF2F Lean4. GPQA is attempted as an optional dataset, but the
official Hugging Face dataset is gated and requires authenticated access.

MATH-500:

```bash
python -m src.evaluation.run_eval \
  --benchmark math500 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --retrieval_source none \
  --use_verifier false \
  --output outputs/math500_qwen.jsonl
```

GSM8K-100 deterministic sample:

```bash
python -m src.evaluation.run_eval \
  --benchmark gsm8k100 \
  --sampling random \
  --sample_seed 0 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --retrieval_source none \
  --use_verifier false \
  --output outputs/gsm8k100_qwen.jsonl
```

AIME 2024:

```bash
python -m src.evaluation.run_eval \
  --benchmark aime2024 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --retrieval_source none \
  --use_verifier false \
  --output outputs/aime2024_qwen.jsonl
```

AIME 2025:

```bash
python -m src.evaluation.run_eval \
  --benchmark aime2025 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --retrieval_source none \
  --use_verifier false \
  --output outputs/aime2025_qwen.jsonl
```

Use `--limit` to run a smaller smoke subset of any preset. If omitted, defaults
are `math=20`, `math500=500`, `gsm8k100=100`, and
`aime2024/aime2025/aime=all available`.

Summaries include overall exact-match accuracy, accuracy by benchmark/category/
level, average runtime, average generated tokens, and verifier statistics:

```bash
python -m src.evaluation.summarize outputs/math500_qwen.jsonl
```

Generic Slurm launcher:

```bash
sbatch scripts/run_benchmark.slurm
sbatch --export=ALL,BENCHMARK=gsm8k100,SAMPLING=random,SAMPLE_SEED=0,OUTPUT_PATH=outputs/gsm8k100_qwen.jsonl scripts/run_benchmark.slurm
sbatch --export=ALL,BENCHMARK=aime2024,OUTPUT_PATH=outputs/aime2024_qwen.jsonl scripts/run_benchmark.slurm
sbatch --export=ALL,BENCHMARK=aime2025,OUTPUT_PATH=outputs/aime2025_qwen.jsonl scripts/run_benchmark.slurm
```

Standard benchmark Slurm scripts:

```bash
sbatch scripts/run_math500_no_retrieval.slurm
sbatch scripts/run_math500_retrieval.slurm
sbatch scripts/run_math500_retrieval_verifier.slurm
sbatch scripts/run_aime2024_baseline.slurm
sbatch scripts/run_aime2025_baseline.slurm
sbatch scripts/run_gsm8k100_baseline.slurm
```

These write JSONL files under `$OUTPUT_DIR/benchmarks`, which is
`/scratch/gilbreth/pham156/math-reasoning-agent/outputs/benchmarks` after
`source scripts/setup_env.sh`.

Compare benchmark runs:

```bash
python -m src.evaluation.compare_runs outputs/benchmarks/*.jsonl
```

The comparison table reports benchmark, mode, number of examples, exact-match
accuracy, average runtime, average generated/input tokens, and verifier valid
rate when verifier rows are present. For MATH-500 it also prints accuracy by
type and level.

Deterministic benchmark-quality rerun:

```bash
sbatch scripts/run_det_math500_no_retrieval.slurm
sbatch scripts/run_det_math500_retrieval.slurm
sbatch scripts/run_det_math500_retrieval_verifier.slurm
sbatch scripts/run_det_gsm8k100_baseline.slurm
sbatch scripts/run_det_aime2024_baseline.slurm
sbatch scripts/run_det_aime2025_baseline.slurm
```

These use deterministic decoding (`temperature=0`, `top_p=1`) and longer
generation limits: 2048 tokens for MATH-500/GSM8K-100 and 4096 tokens for
AIME 2024/2025. Outputs are saved under:

```text
$OUTPUT_DIR/benchmarks_deterministic/
```

Summarize the clean rerun:

```bash
python -m src.evaluation.compare_runs outputs/benchmarks_deterministic/*.jsonl
```

### Retrieval Gating

Analyze selective retrieval policies on MATH-500 using the deterministic baseline
and always-retrieval runs:

```bash
python -m src.evaluation.retrieval_gating
```

This writes per-policy JSONL files plus:

```text
$OUTPUT_DIR/retrieval_gating/summary.json
$OUTPUT_DIR/retrieval_gating/summary.md
```

Supported policies:

- `none`: no retrieval baseline
- `always`: always use full retrieval context
- `category_gate_oracle`: use retrieval for analysis-derived helpful categories
- `category_gate_keyword`: legacy lightweight keyword category predictor
- `category_gate_predicted_0.50`, `0.60`, `0.70`, `0.80`: few-shot LLM category predictor with confidence fallback to no retrieval
- `similarity_gate_0.70`, `0.73`, `0.76`, `0.79`
- `compact_retrieval`: use compact strategy hints without full retrieved solutions

Generate few-shot LLM category predictions, then evaluate the confidence
threshold sweep:

```bash
python -m src.evaluation.category_classifier \
  --benchmark math500 \
  --model_path "$PROJECT_ROOT/models/qwen2.5-math-7b-instruct" \
  --output outputs/retrieval_gating/category_predictions_llm.jsonl

python -m src.evaluation.retrieval_gating \
  --category_predictions outputs/retrieval_gating/category_predictions_llm.jsonl
```

Or run the classifier plus gating summary as one Slurm job:

```bash
sbatch scripts/run_gating_llm_category.slurm
```

Run the compact retrieval model job and refresh the gating summary:

```bash
sbatch scripts/run_gating_compact_retrieval.slurm
```

### Verifier-Guided Best-of-N

Use verifier-guided selection to sample multiple independent compact-retrieval
solutions, score each candidate with the verifier, and keep the highest-scoring
candidate. Start with MATH-100 before scaling to MATH-500:

```bash
sbatch scripts/run_math100_baseline.slurm
sbatch scripts/run_math100_compact_retrieval.slurm
sbatch scripts/run_math100_compact_best_of_3.slurm
sbatch scripts/run_math100_compact_best_of_5.slurm
sbatch scripts/run_math100_compact_consensus_best_of_3.slurm
sbatch scripts/run_math100_compact_consensus_best_of_5.slurm
```

The best-of runs use:

```bash
python -m src.evaluation.run_eval \
  --benchmark math500 \
  --limit 100 \
  --retrieval_source math \
  --retrieval_prompt_style compact \
  --num_candidates 3 \
  --selection consensus_verifier \
  --temperature 0.7 \
  --top_p 0.95 \
  --output outputs/verifier_best_of_n/math100_compact_consensus_best_of_3.jsonl
```

Each JSONL row stores `candidate_solutions`, per-candidate verifier scores,
`selected_candidate`, `selection_metadata`, `final_extracted_answer`, answer
groups, selected answer group, and oracle candidate availability. Consensus
selection groups candidates by normalized final answer, chooses a majority
answer group when one exists, then chooses the highest verifier score within
that group. If no majority exists, it falls back to verifier-only selection.
Compare completed runs:

```bash
python -m src.evaluation.compare_best_of_n outputs/verifier_best_of_n/math100_*.jsonl
```

Scale the consensus experiment to MATH-500 under the 4-hour standby limit by
running best-of-5 jobs as five 100-example array shards:

```bash
sbatch scripts/run_math500_consensus_baseline.slurm
sbatch scripts/run_math500_consensus_compact.slurm
sbatch scripts/run_math500_consensus_no_retrieval_best_of_5_array.slurm
sbatch scripts/run_math500_consensus_compact_best_of_5_array.slurm
```

After those complete, merge shards and compare fallback choices:

```bash
sbatch scripts/run_math500_consensus_merge_analyze.slurm
```

The fallback analysis recomputes no-majority behavior from the saved candidates:
`verifier`, `first`, and `abstain`. It writes:

```text
outputs/consensus_math500/fallback_analysis.json
outputs/consensus_math500/fallback_analysis.md
outputs/consensus_math500/compare_best_of_n.md
```

## Future Web UI Plan

Use FastAPI for the backend:

- `/solve`: run retrieval, generation, verification, revision
- `/retrieve`: inspect nearest solved examples
- `/verify`: verify a proposed solution
- `/trace/{run_id}`: return candidate, verifier, and revision trace

Use React for the frontend:

- problem input
- retrieved examples panel
- solution timeline
- verifier issues panel
- later proof-tree/search visualization with expandable nodes

Keep the web UI separate from the training path so inference remains usable on Slurm or interactive GPU sessions.

## Checkpointed Algorithm Pilots

The four GPU smoke tests passed. The next stage runs correctness-only GRPO, Dr. GRPO, DAPO-style optimization, and GSPO from Qwen2.5-Math-1.5B base, seed 0. Each pilot uses the same 512 MATH Level 3-5 problems, four responses per prompt, 128 optimizer updates (one full pass), LR 1e-6, and a 3,000-token completion cap. Preserve the tested full FP32 trainable/master parameters with BF16 compute. This is a documented precision adjustment from the original pure-BF16 proposal.

```bash
cd /home/pham156/math-reasoning-agent
source scripts/setup_env.sh
# Create once. Subsequent jobs reuse this immutable split.
bash scripts/rl/smoke_python.sh -m src.training.rl.pilot --prepare_splits \
  --output_dir "$OUTPUT_DIR/rl_1p5b_study/split_preflight"
bash scripts/rl/submit_algorithm_pilots.sh --dry-run
# Queue only when ready to spend the pilot budget:
bash scripts/rl/submit_algorithm_pilots.sh --submit
```

Split creation audits the local MATH train dataset for parseable reference answers, prompt length, and duplicate problem hashes. It first reserves 300 category/level-balanced development examples from eligible Levels 3-5, then selects 512 training examples using split seed 42. Files live in `$DATA_DIR/rl_1p5b_pilot_v1/`. The pilot evaluates a fixed 20-example slice before/after training; these diagnostics do not replace the planned 100/300-example development evaluations or final benchmarks. Do not rerun `--prepare_splits` once the directory exists.

The submission matrix is four restart tests followed by up to three four-hour pilot segments per method, with at most two simultaneous GPUs and `ai-forge/a100-80gb/standby` throughout. All four restart checks must pass before any pilot starts. Each check compares eight uninterrupted updates against four updates plus a fresh-process checkpoint continuation, requiring identical sampled rollouts and matching saved weights within 1e-7. Checkpoint equivalence remains unverified until those GPU checks complete.

Replacement runs enable Transformers `full_determinism` and PyTorch deterministic algorithms before model initialization, with deterministic cuBLAS workspace settings, cuDNN benchmarking disabled, and TF32 disabled. Random sampling remains enabled during RL, using reproducible random state. The check first compares both checkpoint-4 snapshots and the first 64 rollouts before starting the continuation, then compares the final eight-update models and all 128 rollouts. A seed alone did not make the previous GPU executions identical; those restart checks failed and the dependent pilots were cancelled. The original smoke tests checked updates/checkpoint saving, rather than restart equivalence. Deterministic training can reduce throughput; the GPU checks still need to establish that continuation works.

Each segment asks the trainer to stop after 2.5 hours at the next complete rollout boundary, saves policy/optimizer/scheduler/RNG/Trainer state, and records `progress.json`. A continuation reloads that checkpoint and skips already consumed batches. Checkpoints with pending rollout slices are rejected. The wall-time reserve is best effort: GPU rollout time varies, and an unexpected timeout or failed restart requires investigation. A completed run makes later segments no-ops; a run still paused after segment three needs another continuation. Dependencies use `afterok`, so failures block further work. The rotating checkpoint set, full final models, rollouts, results, and submission log remain under `$OUTPUT_DIR/rl_1p5b_study/pilot_<timestamp>/`; small Slurm logs use `%A_%a_%x` in home. The 5,000-example/three-seed and reward-design grids are deferred until these pilots establish runtime and stability. A final dependency job writes `pilot_summary.json` and `pilot_summary.md`, including baseline/after correctness, fixes, breaks, response length, truncation, and training metrics.

### Shared Development Evaluation and SFT Control

`src.evaluation.pilot_dev` evaluates all 300 frozen development problems with the exact pilot chat prompt, greedy decoding, a 3,000-token cap, FP32 weights/BF16 compute, and the pilot exact/Math-Verify grader. It writes flushed per-example JSONL and a summary under scratch; rerunning resumes completed rows. `scripts/rl/run_pilot_dev.slurm` evaluates the frozen baseline and four RL final models as five independent array tasks without an artificial concurrency limit. Set `RL_STUDY_ROOT` to the completed pilot root and `RL_COMPARISON_ROOT` to a new scratch comparison directory before submission.

Development evaluation defaults to `--segment_seconds 12600` (3.5 hours), pausing between examples to leave room before the four-hour limit; individual example latency can vary. Rerun the same job to append only unfinished examples. Chain continuation jobs only for that model; completed runs are no-ops. A successful segment may still be partial: require all 300 rows and the model summary before treating evaluation as complete. The combined comparison rejects incomplete results.

Pilot figures are generated by `src.evaluation.plot_pilot_study --pilot_root <completed-pilot-root> --comparison_root <development-comparison-root>`. Use the same Apptainer wrapper with `RL_SMOKE_PYTHON=$SCRATCH_ROOT/env/rl-plots/bin/python`; this separate scratch venv needs `matplotlib==3.10.7` and does not modify the training runtime. It exports a training dashboard (reward, response length, mixed-reward groups, truncation, entropy, SFT loss), completed development accuracies, PNG/PDF figures, and small raw metric CSV under `$PROJECT_ROOT/figures/<comparison-name>/` in home. Override with `--output_dir` if needed. Models, datasets, checkpoints, evaluation JSONL, and plotting caches stay in scratch. The accuracy chart excludes incomplete evaluations; rerun after baseline/SFT complete. KL is absent because it was not logged.

`scripts/rl/check_sft_termination.slurm` runs a five-problem diagnostic for the original SFT model without changing it or its existing evaluation. `src.evaluation.check_sft_termination` audits all 512 supervised response endings, saves raw generated token IDs, and compares original greedy outputs with the prefix ending at the first chat-end token. Results are in `<comparison-root>/termination_diagnostic/summary.json`. This isolates whether inference ignores an emitted chat-end marker; if the model never emits it, adding a stop token alone cannot fix repetition. Training/evaluation stopping conventions must be aligned before treating SFT as a clean comparison.

### RL Evaluation Controls and Exploration

`src.evaluation.pilot_exploration` uses both `endoftext` and `im_end` as accepted stop tokens for every model. It preserves the pilot prompt, 3,000-token cap, exact/Math-Verify grader, FP32 parameters/BF16 compute, and frozen development split. Corrected greedy controls cover all300 problems for baseline, GRPO, Dr. GRPO, DAPO-style, GSPO, and the original SFT control. Sampled diagnostics cover exactly the first100 development problems for baseline and the four RL models, generating eight responses per prompt in one inference batch at temperature1/top-p1/top-k0. SFT is excluded from the sampled grid. These results are separate from original endoftext-only evaluations; no model training is changed.

Set `RL_DIAGNOSTIC_ROOT` to a new scratch directory. `scripts/rl/run_common_stop_controls.slurm` has18 independent tasks (six models x three100-problem shards); `scripts/rl/run_pilot_exploration.slurm` has25 independent tasks (five models x five20-problem shards). No artificial concurrency limit or dependency exists between these arrays. Each task flushes completed problems to JSONL, pauses between problems after3.5hours, and can resume the same shard. Sample seeds depend on source ID rather than shard execution order. Shard configuration, stop IDs, split IDs and exact prompt hashes are checked on resume; `--prepare_only` audits tokenizers and disjoint splits on CPU. Investigate or continue incomplete shards before interpreting results.

After both arrays finish, `scripts/rl/summarize_pilot_exploration.slurm` compares original/common-stop greedy accuracy and fixes/breaks, then reports sampled pass@1/4/8, mixed/all-correct/all-wrong groups, length and truncation. It refuses incomplete or mismatched data. The pass@k estimator is `1-C(8-c,k)/C(8,k)`; sampled pass@1 is not greedy accuracy. Gained sampled success does not establish newly discovered capability: zero successes in eight baseline samples is only finite-sampling evidence. Raw outputs and summaries stay scratch; plots/exportable figures belong in home. This is a single-training-seed development diagnostic, not final test evaluation or hyperparameter tuning.

Export completed results with the isolated plotting runtime: `RL_SMOKE_PYTHON=$SCRATCH_ROOT/env/rl-plots/bin/python bash scripts/rl/smoke_python.sh -m src.evaluation.plot_pilot_exploration "$RL_DIAGNOSTIC_ROOT"`. PNG/PDF dashboards land in `$PROJECT_ROOT/figures/<diagnostic-root-name>/` in home; plotting caches remain scratch.

### Controlled Learning-Rate Pilots

`src.training.rl.pilot --learning_rate` makes LR configurable; default1e-6 preserves historical behavior. Resume rejects LR changes, interpreting older checkpoint arguments without this field as1e-6. The learning-rate study compares2e-7/5e-7/1e-6 for each of four methods on the same512 frozen train problems, seed0,128 updates, four responses/problem, correctness-only reward, and unchanged training generation/stopping behavior. It reuses the four completed1e-6 models and their common-stop dev300 evaluations. Only eight new pilots train, always from the untouched base model. Training-internal20-problem diagnostics remain historical; tuning decisions use the separate common-stop dev300 results.

Run `/usr/bin/python3 scripts/rl/submit_lr_pilots.py --root "$OUTPUT_DIR/rl_1p5b_study/lr_tuning_20261014"` for validation and Slurm dry runs; add `--submit` to launch. This submission utility uses only the host standard library to access Slurm; all training/evaluation runs use the isolated scratch Apptainer runtime. Every new run has four independent checkpoint-continuation segments (afterok within that experiment only) followed by a three-shard common-stop evaluation. Already completed segments become no-ops. There is no artificial concurrency limit or barrier across independent methods/rates. Final evaluation refuses incomplete training; if a run remains paused after four segments, continue it before resubmitting its evaluation. The manifest is saved after every submission to prevent accidental duplicate launches.

`src.evaluation.summarize_lr_pilots <tuning-root>` combines the12 settings, reporting accuracy, length, truncation, reward/mixed-group statistics, fixed/broken problems, memory, and job IDs for total Slurm cost accounting. Last-segment trainer runtime is not total training cost. Provisional per-method settings maximize development accuracy, with lower generated tokens then lower LR as tie-breaks; confirm promising settings with another seed before main-study selection. No MATH-500/AIME test data is used for tuning. All artifacts remain scratch except plots/exportable figures in home; logs are ID-first. These are LR-only pilots, not a rollout/clipping/reward sweep.

### Non-Math Retention Check

The frozen base and four completed seed0 RL pilots are compared on32 seeded nonoverlapping512-token [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext) test blocks,100 [BoolQ](https://huggingface.co/datasets/google/boolq) validation questions, and100 [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) test questions. Dataset revisions, sampled IDs, tokenized inputs, and vocabulary hash are frozen once under `$DATA_DIR/rl_retention_v1/frozen.json`, seed42. This is a small retention diagnostic; it is never an LR/checkpoint-selection score and is not proof of broad ability preservation or uncontaminated pretraining data.

Run `bash scripts/rl/smoke_python.sh -m src.evaluation.retention --prepare` on CPU to download into scratch and freeze inputs. All downloads/caches stay scratch. Set `RL_RETENTION_ROOT=$OUTPUT_DIR/rl_1p5b_study/retention_20261014`, then submit `scripts/rl/run_retention.slurm`: five independent, unthrottled standby tasks. The output-root parent is the existing `rl_1p5b_study` folder containing the completed `pilot_20261013_000313` models. The evaluator reuses FP32 parameters/BF16 compute, writes per-item JSONL, and resumes completed items; a3.5hour safe pause leaves room before the4hour allocation ends. Submit `scripts/rl/summarize_retention.slurm` after successful completion of the array.

Language loss scores511 next-token targets per512-token block without chat formatting; perplexity is `exp(total NLL/scored tokens)`. This is subset perplexity with reset block contexts, not the full WikiText-2 benchmark number. BoolQ/ARC choices use zero-shot conditional log likelihood of single-token option letters, under both raw question/answer and neutral chat templates. This measures label selection, not free-form generation, coding, instruction-following compliance, or proof quality; it has no stop-token dependence. The frozen base tokenizer must match each model vocabulary exactly. No examples are silently truncated. Summaries report paired deltas/fixes/breaks, label counts and majority baselines, plus paired item/block bootstrap intervals (not training-seed uncertainty or multiplicity-corrected significance). Positive delta language NLL and negative task accuracy changes indicate possible forgetting; confirm apparent regressions on larger/independent checks before broad conclusions.

### External Math Benchmarks For The 1.5B Pilots

Freeze MATH-500 test (all500) and a seed42 random GSM8K test subset (100) with `bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_benchmarks "$RL_BENCHMARK_ROOT" --prepare`, where `RL_BENCHMARK_ROOT=$OUTPUT_DIR/rl_1p5b_study/external_benchmarks_20261015`. Preparation records source metadata and exact inputs, rejects duplicate/empty-answer problems and normalized-text overlap with the512 RL training problems, and refuses changes to existing frozen inputs. These benchmarks were inspected during historical7B work, so they are external transfer checks, not untouched confirmation sets; pretraining contamination is unknown.

Submit `sbatch --export=ALL,RL_BENCHMARK_ROOT="$RL_BENCHMARK_ROOT" scripts/rl/run_external_benchmarks.slurm`. Thirty independent unthrottled standby tasks evaluate the frozen base and four original1e-6/128-update RL models: five100-problem MATH-500 shards plus one100-problem GSM8K shard per model. The common evaluator preserves FP32 parameters/BF16 compute, exact training chat prompts, greedy decoding,3000-token cap, endoftext/im_end stopping, and exact/Math-Verify grading. No retrieval, verifier, sampling selection, or new training. JSONL/configs stay scratch; logs are job-ID-first in home. The evaluator safely pauses after3.5hours; rerun any incomplete shard with the same environment/array index before summarizing.

Submit `scripts/rl/summarize_external_benchmarks.slurm` with the same export and `--dependency=afterok:ARRAY_JOB_ID`. It rejects incomplete shards or changed benchmark/prompt hashes and writes `summary.{json,md}` under the output root: accuracy, tokens, runtime, truncation, paired fixes/breaks, and type/level breakdowns. Do not use these scores for learning-rate/checkpoint selection. Results are single-training-seed diagnostics, not final multi-seed method rankings.

### Lean Formal-Proof Feasibility Run

This extension performs inference only: untouched Qwen2.5-Math-1.5B base,20 miniF2F validation problems,8 sampled tactic proofs each, temperature1/top-p1/top-k0, completion cap2048, and common endoftext/im_end stopping. No RL updates, reference solutions, theorem retrieval, or proof demonstrations are used. The [miniF2F mirror](https://huggingface.co/datasets/cat-searcher/minif2f-lean4) revision and source IDs are frozen under `$DATA_DIR/lean_smoke_v1/`. Its13 commented-out/error statements are recorded rather than repaired. Checker preflight selects the first20 statements that typecheck in seed42-shuffled validation order, recording compatibility exclusions. This is a restricted, compatibility-screened feasibility subset, not a full miniF2F benchmark result.

Lean4.24.0 and Mathlibv4.24.0 (commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`) are installed under `$SCRATCH_ROOT/lean/`. The helper container is `$SCRATCH_ROOT/containers/lean-tools.sif`; it supplies git/curl/runtime libraries, while the compiler is explicitly pinned via PATH. All Apptainer, Elan, Lake and Mathlib caches stay scratch. `scripts/lean/lean.sh --setup <command>` permits setup writes; proof checking uses read-only mounts, hides home/cwd, and exposes only the Lean runtime and candidate sources. No host packages are installed. Public compiler release: `https://github.com/leanprover/lean4/releases/download/v4.24.0/lean-4.24.0-linux.tar.zst`; Mathlib cache installation uses `bash scripts/lean/lean.sh --setup lake exe cache get`. Set Apptainer cache/tmp to scratch before pulling helper images.

Prepare the statement pool with `bash scripts/rl/smoke_python.sh -m src.evaluation.lean_smoke --prepare --output_dir "$LEAN_SMOKE_ROOT"`. `scripts/lean/preflight.slurm` packs the installed runtime into a read-only SquashFS image to avoid per-file scratch metadata latency, then runs positive, deliberately unfinished, and incorrect-proof controls. It freezes the20 compatible statements only after controls pass. Generation must depend on successful preflight; never bypass a failed checker gate. `scripts/lean/run_smoke.slurm` uses four independent shards of five problems. GPU generation runs inside the existing Python container; the host standard-library driver then starts the separate Lean container, avoiding unsupported nested Apptainer execution. `scripts/lean/summarize_smoke.slurm` follows the generation/checking array. All jobs use standby, ID-first logs and scratch results.

Only tactic bodies starting with `by` are accepted. The checker rejects sorry/admit, extra declarations, selected metaprogramming constructs, comments/quoted code and native_decide. It checks the unchanged formal goal, enforces4GiB Lean memory/200000 heartbeat and180second process limits, and audits `#print axioms candidate_proof` against propext/Classical.choice/Quot.sound. This is a conservative proof-submission policy, not a certified security sandbox; valid proofs outside the supported syntax may be rejected. Proof correctness is relative to the formal statement and trusted library/axioms, not an automatically checked English translation. Reports retain raw generations, extracted proofs, errors, timeouts, checking time, truncation, verified-attempt counts and observed success within8 attempts. Zero successes would indicate insufficient signal for direct binary-reward Lean RL under this setup, not an impossibility result.

`src.training.rl.sft_pilot` trains the same Qwen2.5-Math-1.5B base on the exact 512 frozen RL training IDs, retrieving their ground-truth solutions from MATH train. It uses the inference chat prefix verbatim, assistant-only cross entropy, one epoch, LR 1e-5, batch size 1, accumulation 4 (128 updates), full FP32 parameters/BF16 compute, and a 4,096-token total cap. Preparation fails on oversized examples rather than silently changing shared data. This is a matched-data control, not a matched-compute comparison. `--prepare_only` audits the data on CPU. Training checkpoints, full final model, loss history, and configuration stay in scratch.

Submit `scripts/rl/run_sft_pilot.slurm` independently; only `scripts/rl/run_sft_dev.slurm` depends on its successful completion. Run `scripts/rl/compare_pilot_dev.slurm` after all six evaluations finish. It writes `evaluation/comparison.{json,md}` with accuracy, fixes/breaks against the frozen baseline, runtime, tokens, and truncation. Every GPU job uses standby and ID-first home logs. These single-seed development results do not replace final test benchmarks.
