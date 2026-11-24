# Verifiable-Reward RL for Mathematical Reasoning

Controlled post-training experiments on Qwen2.5-Math-1.5B: GRPO, Dr. GRPO, a DAPO-style objective variant, and GSPO, with no-RL and supervised fine-tuning controls. The study examines final-answer accuracy, exploration, optimization, reward design, inference-time selection, and generalization. The repository also preserves earlier 7B retrieval and verifier experiments separately from the matched 1.5B study.

## Reports and Results

| Document | PDF | Editable source |
|---|---|---|
| Midterm update | [Two-page IEEE report](docs/midterm/midterm_update.pdf) | [LaTeX](docs/midterm/main.tex) |
| Completed-results atlas | [All completed comparisons](docs/results_compendium/completed_results.pdf) | [LaTeX](docs/results_compendium/main.tex) |
| Final report draft | [IEEE draft](docs/final/final_report.pdf) | [LaTeX](docs/final/main.tex) |
| A0 poster draft | [Poster PDF](docs/poster/research_poster.pdf) | [SVG](docs/poster/research_poster.svg), [preview](docs/poster/research_poster.png) |

The documents are dated snapshots, not interchangeable summaries: the atlas includes more completed conditions than the earlier report drafts. Completed negative results are retained. Failed or partial benchmark cohorts are not promoted to full-benchmark scores. Final-answer correctness does not certify proof validity.

## Implementation

See [the implementation and ablation guide](docs/implementation.md) for the code-to-experiment map, entry points, and tests.

- `src/training/rl/`: objective configuration, RL training, reward variants, continuation, checkpoint management, and SFT/RFT controls.
- `src/evaluation/`: shared grading, benchmark evaluation, exploration, paired comparisons, and checkpoint-only ablations.
- `scripts/paper/`: evidence validation, tables, plotting, and document generators.
- `containers/`: pinned Apptainer runtime recipe and dependencies.
- `tests/`: objective, reward, restart, evaluation, ablation, and publication checks.

GRPO/Dr. GRPO/DAPO-style/GSPO are implemented experimental arms. DAPO-style is not the full DAPO training recipe: dynamic sampling is not enabled. Classical PPO/REINFORCE modules are research scaffolding, not completed study results. The current 1.5B RL experiments use full-parameter updates, not the older 7B QLoRA workflow.

## Gilbreth Setup

```bash
source scripts/setup_env.sh
bash scripts/rl/setup_smoke_runtime.sh
bash scripts/rl/smoke_python.sh -m scripts.download_data
bash scripts/rl/smoke_python.sh -m scripts.download_benchmarks --help
```

The setup derives the code directory from the checkout and defaults large storage to `/scratch/gilbreth/$USER/math-reasoning-agent`. Set `SCRATCH_ROOT` before sourcing to use another directory under `/scratch/gilbreth`. Weights, datasets, environments, caches, raw outputs, and checkpoints stay in scratch. Small logs and curated report/figure exports stay in home. Runtime setup installs dependencies only in the isolated scratch container environment; it also downloads the 1.5B base model. Do not run training on a login node.

The primary source is MATH train; the completed pilot uses 512 Level 3-5 training problems and 300 disjoint development problems from that same source split. Dev100 is the fixed first 100 development problems, not a separate benchmark. MATH-500 is from MATH test. The atlas documents split provenance, filtering, and exact normalized-text overlap checks. GSM8K and pinned AIME sources have separate evaluation protocols; the two AIME dataset providers are not silently pooled.

Inspect supported commands before running:

```bash
bash scripts/rl/smoke_python.sh -m src.training.rl.pilot --help
bash scripts/rl/smoke_python.sh -m src.evaluation.pilot_exploration --help
bash scripts/rl/smoke_python.sh -m src.evaluation.checkpoint_ablations --help
bash scripts/rl/smoke_python.sh -m unittest discover -s tests
```

Training and evaluation require the relevant scratch models/data and an allocated GPU. Restart gates must pass before interrupted training is used as a reproducible continuation. Cluster-specific Slurm files and job manifests are intentionally excluded from the public repository; some historical launcher helpers therefore require local cluster scripts. Python training/evaluation implementations and container recipes are included.

## Rebuilding Documents

LaTeX sources and their referenced figures/tables are included, so report editing does not require the private experiment output tree. Compile each `main.tex` using its documented engine, or upload its source folder to Overleaf. ZIP exports are not versioned. The poster is also supplied as an editable vector SVG.

Regenerating numerical tables or plots from scratch is different from compiling a manuscript: the generators require the original saved experiment evidence and validate its provenance. Model weights, raw candidate JSONL, and checkpoints are not distributed. Existing summaries do not replace those source files when a generator requires them.

[Earlier workflow documentation](docs/legacy_workflows.md) preserves the original retrieval, QLoRA, benchmark, and cluster command references. Those historical instructions are not the current matched RL protocol.
