# Algorithm and Ablation Implementation

## Core Algorithms

| Component | Implementation | Validation |
|---|---|---|
| Objective registry | `src/training/rl/methods.py` selects and checks TRL loss, reward scaling, importance-sampling level, and clipping | `tests/test_rl_smoke.py`, `tests/test_rl_objectives.py` |
| RL trainer | `src/training/rl/smoke.py` configures GRPOTrainer, explicit objective settings, deterministic sampling, logged reward components, and safe stop boundaries | `tests/test_rl_smoke.py`, `tests/test_rl_pilot.py` |
| Four-arm pilot | `src/training/rl/pilot.py` creates frozen training/development splits and validates training/resume arguments | `tests/test_rl_pilot.py` |
| Correctness reward | `src/training/rl/smoke.py` and `src/evaluation/answer.py` extract answers and grade with normalized exact/Math-Verify matching | `tests/test_rlvr_rewards.py`, `tests/test_answer_extraction.py`, `tests/test_regrade.py` |
| Restart checks | `src/evaluation/check_rl_resume.py` compares boundary state, resumed weights, and sampled completions | `tests/test_rl_pilot.py`, `tests/test_grpo_extension.py` |

The registry configures the active TRL objectives; `objectives.py` additionally provides small tensor-level reference functions, not a replacement production trainer. GRPO uses group-scaled advantages and response-normalized loss; Dr. GRPO disables group reward scaling and uses a fixed completion-length denominator. DAPO-style uses group scaling, asymmetric token-level clipping, and active-token normalization, including a trainer adapter for accumulated batches. GSPO uses sequence-level importance ratios. Neither DAPO dynamic sampling nor a complete paper-scale reproduction is claimed.

Historical lenient reward grading and strict closed-box reporting are separate APIs. The merged extraction fixes and `src/evaluation/regrade.py` are included, but do not retroactively redefine saved training rewards or published scores. Existing historical checkpoints/results must be evaluated with the corresponding recorded grader and source hashes.

## Training-Time Ablations

| Factor | Implementation | Tests |
|---|---|---|
| Learning rate | `src/training/rl/pilot.py --learning_rate`; `src/evaluation/summarize_lr_pilots.py` | `tests/test_lr_pilot.py`, `tests/test_lr_grid.py` |
| KL coefficient and clipping bounds | `src/training/rl/ablation_controls.py`; pilot `--kl_beta`, `--clip_lower`, `--clip_upper`; `src/evaluation/summarize_grpo_ablations.py` | `tests/test_grpo_ablations.py` |
| Correctness, format bonus, length penalty | `src/training/rl/reward_design.py`; frozen settings in `study_controls.py`; `src/evaluation/summarize_grpo_studies.py` | `tests/test_grpo_studies.py` |
| Rollout temperature and group size | `pilot.py --temperature/--num_generations`, `study_controls.py`; shared reward-group statistics | `tests/test_grpo_studies.py` |
| Additional training passes | `src/training/rl/extend.py` forks continuation without mutating controls; epoch-addressed sampler in `smoke.py` | `tests/test_grpo_extension.py` |
| Storage and continuation safety | `src/training/rl/checkpoint_storage.py` validates retention, complete resume state, and cleanup receipts | `tests/test_checkpoint_storage.py` |
| Human-target SFT and correct-solution RFT | `src/training/rl/sft_pilot.py`, `rft_data.py`; `src/evaluation/summarize_sft_study.py` | `tests/test_sft_pilot.py`, `tests/test_sft_aime.py` |

Learning rate, reward shaping, sampling, clipping, and KL controls are distinct interventions; completed one-factor results do not imply every combination was tested. SFT is a supervised control, not RL. RFT filters generated solutions by final-answer reward before supervised training; that filter does not certify every reasoning step.

## Evaluation and Checkpoint-Only Ablations

| Analysis | Implementation | Tests |
|---|---|---|
| Greedy and sampled evaluation; pass@k | `src/evaluation/pilot_exploration.py`, `summarize_pilot_exploration.py` | `tests/test_pilot_exploration.py` |
| MATH-500/GSM8K and frozen benchmark rows | `src/evaluation/pilot_benchmarks.py`, `checkpoint_benchmarks.py`, `external_eval.py` | `tests/test_pilot_benchmarks.py`, `tests/test_checkpoint_benchmarks.py` |
| AIME 2025/2026 | `src/evaluation/aime_benchmarks.py`; provider-specific loading in `benchmarks.py` | `tests/test_aime_setup.py` |
| Token budget and inference temperature | `src/evaluation/checkpoint_ablations.py`; validated decoding overrides in `pilot_exploration.py` | `tests/test_checkpoint_ablations.py`, `tests/test_pilot_exploration.py` |
| Prompt sensitivity and external consensus | `src/evaluation/checkpoint_followup.py` | `tests/test_checkpoint_followup.py` |
| Candidate budget, majority/plurality rules, diversity, subgroup fixes/breaks | `src/evaluation/saved_candidate_ablations.py` | `tests/test_checkpoint_followup.py` |
| Replayed checkpoint evaluation | `src/training/rl/dense_replay.py`, `src/evaluation/dense_curve.py` | `tests/test_sft_aime.py`, `tests/test_grpo_extension.py` |
| General-task retention | `src/evaluation/retention.py`, `summarize_retention.py` | `tests/test_retention.py` |

Candidate-subset/order averages are retrospective analyses, not additional independently generated runs. pass@k measures oracle availability, not a deployable selector. Greedy accuracy, sampled pass@1, majority-selected accuracy, and strict extraction accuracy have different denominators/protocols and remain separately labeled.

## Reports and Plotting

`scripts/paper/plot_midterm.py`, `prepare_final.py`, `prepare_poster.py`, and `prepare_compendium.py` create the four document bundles. `compendium_setup.py`, `compendium_layout.py`, and `compendium_supplement.py` organize setup, numerical comparisons, and detailed diagnostics. `refresh_completed.py` only promotes complete evaluated cohorts and preserves missing checkpoint gaps. Associated `test_*report.py`, `test_compendium.py`, `test_poster.py`, and `test_completed_refresh.py` cover scope and evidence validation.

Published PDFs and referenced assets are under `docs/{midterm,final,poster,results_compendium}/`. Raw experiment data, local campaign manifests, source-linked runtime evidence, cluster job scripts, and assistant state are not required to read the documents and are not bundled as public training artifacts. The compact atlas ledger is included for its numerical records and provenance; its historical paths are references, not downloadable checkpoints.
