# math-reasoning-agent

An AI-heavy mathematical reasoning agent for competition and proof-style math on Purdue Gilbreth.

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

The old top-level command paths such as `python src/run_pipeline.py` and
`python src/build_index.py` are kept as thin wrappers for compatibility with
the README and Slurm scripts.

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
python src/build_index.py
```

Outputs:

```bash
$DATA_DIR/faiss/math.index
$DATA_DIR/faiss/metadata.jsonl
$DATA_DIR/faiss/index_config.json
```

Test retrieval:

```bash
python src/retrieve.py --problem "Find all integer solutions to x^2 - y^2 = 15." --top_k 3
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
python src/retrieve.py \
  --problem "Find all integer solutions to x^2 - y^2 = 15." \
  --top_k 4 \
  --candidate_k 40

python src/retrieve.py \
  --problem "Prove that there are infinitely many primes." \
  --show_context

python src/retrieve.py \
  --problem "Prove that there are infinitely many primes." \
  --show_solver_prompt
```

Evaluate retrieval quality:

```bash
python src/evaluate_retrieval.py --split test --num_queries 200 --top_k 10
```

This writes:

```bash
$LOG_DIR/retrieval_eval_summary.json
$LOG_DIR/retrieval_eval_rows.csv
```

## Baseline Inference

```bash
python src/infer_baseline.py \
  --problem "Prove that the sum of two even integers is even." \
  --max_new_tokens 1024 \
  --temperature 0.2
```

To compare DeepSeek:

```bash
python src/infer_baseline.py \
  --model_path "$MODEL_DIR/deepseek-math-7b-instruct" \
  --problem "Solve for all real x such that x^2 - 5x + 6 = 0."
```

## End-to-End Agent Pipeline

```bash
python src/run_pipeline.py --problem "Prove that there are infinitely many primes."
```

Pipeline:

1. retrieve similar solved examples from FAISS
2. build a RAG prompt
3. generate a candidate solution
4. verify for algebra mistakes, logical gaps, invalid assumptions, and missing justification
5. revise if the verifier marks the solution invalid or uncertain
6. print the final solution

Control the self-critique loop:

```bash
python src/run_pipeline.py \
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
python src/verifier_agent.py \
  --problem "Solve x^2 - 5x + 6 = 0." \
  --solution "x=2" \
  --print_prompt
```

Run structured verification:

```bash
python src/verifier_agent.py \
  --problem "Solve x^2 - 5x + 6 = 0." \
  --solution "x=2"
```

Run the standalone self-critique loop:

```bash
python src/self_critique_loop.py \
  --problem "Prove that the sum of two even integers is even." \
  --max_iterations 2
```

JSON trace output:

```bash
python src/self_critique_loop.py \
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
python src/proof_search.py \
  --problem "Prove that the sum of two even integers is even." \
  --max_depth 2 \
  --beam_width 3 \
  --branch_factor 2
```

Use retrieval context and verifier scoring:

```bash
python src/proof_search.py \
  --problem "Prove that sqrt(2) is irrational." \
  --use_retrieval \
  --use_verifier \
  --max_depth 2 \
  --beam_width 3
```

Export the tree as JSON:

```bash
python src/proof_search.py \
  --problem "Prove that there are infinitely many primes." \
  --format json \
  --output proof_search_primes.json
```

Export Mermaid for quick visualization:

```bash
python src/proof_search.py \
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
python src/finetune_qlora.py --max_train_samples 32 --save_steps 20
```

Full conservative first pass:

```bash
python src/finetune_qlora.py
```

Adapter output:

```bash
$OUTPUT_DIR/qwen2.5-math-7b-math-lora
```

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

All Slurm logs are written to:

```bash
$LOG_DIR/
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
python src/evaluate_agent.py \
  --split test \
  --limit 20 \
  --modes baseline,retrieval,verifier,search \
  --experiment_name smoke_eval
```

Run only cheaper modes:

```bash
python src/evaluate_agent.py \
  --split test \
  --limit 50 \
  --modes baseline,retrieval \
  --experiment_name baseline_vs_rag
```

Compare verifier statistics for every mode output:

```bash
python src/evaluate_agent.py \
  --split test \
  --limit 20 \
  --modes baseline,retrieval,verifier,search \
  --verify_all_outputs \
  --experiment_name verified_comparison
```

Run search with retrieval and verifier scoring:

```bash
python src/evaluate_agent.py \
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
