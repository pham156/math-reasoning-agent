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

### Data and Rollout Flow

`pilot.py` filters MATH-train examples by difficulty, reference-answer parseability, prompt length and normalized-text duplication. It reserves the stratified development split before selecting training examples, then saves the actual source indices and hashes. The completed main study uses 512 training problems and 300 disjoint development problems, with a fixed 100-problem development subset for sampled diagnostics. These are not slices of MATH-500.

Each RL example contains the system/user conversation and reference answer. The human solution is not a supervised target for RL. `smoke.py` renders the tokenizer's chat template and samples four responses per problem by default. Sixteen prompts yield 64 responses; accumulation consumes 16 responses per optimizer update, so one rollout batch supplies four updates. `num_iterations=1` does not mean one update per entire rollout batch. The generation and optimizer batches are recorded separately to make this accounting explicit.

### Reward Computation and Policy Loss

`grade_solution` extracts a final answer and first attempts whitespace-insensitive equality. If that fails, Math-Verify parses the reference and prediction as mathematical answers and tests equivalence; empty answers and parsing/verification exceptions receive zero. This outcome reward does not inspect proof validity, execute generated code, or use the older verifier agent. `reward_design.py` optionally adds 0.10 for exactly one nonempty parseable boxed answer or subtracts up to 0.05 in proportion to completion length. The callback logs raw correctness separately from shaping, saves each sampled completion, and aggregates within-prompt reward statistics.

TRL computes the clipped policy-gradient surrogate from current/old-policy likelihood ratios and group-relative advantages. `methods.py` explicitly fixes reward scaling, ratio level, loss normalization and clipping bounds, then checks those settings against the constructed trainer. GRPO divides centered rewards by group reward standard deviation and averages length-normalized response losses. Dr. GRPO removes group-standard-deviation scaling and uses a fixed maximum-length loss denominator. DAPO-style instead normalizes over active batch tokens; the trainer adapter corrects that denominator when a generation batch spans several accumulated updates. GSPO replaces token-level importance ratios with a sequence ratio derived from the mean token log-ratio. Tensor helpers in `objectives.py` are unit-test references, not the production implementation of every method.

### Parameters, Numerical Precision and Resume State

The main 1.5B RL study updates all policy parameters, not LoRA adapters. Trainable weights remain FP32, while BF16 autocast and gradient checkpointing reduce compute memory. AdamW uses a constant learning-rate schedule in these RL runs. Nonzero-KL arms require an explicitly frozen reference policy initialized from the same base model; the trainer verifies reference precision and records measured KL. The zero-KL arms do not allocate this additional reference model. A callback rejects nonfinite optimization metrics.

Resuming loads the Trainer checkpoint, including model, optimizer, scheduler, trainer and RNG state; loading exported model weights alone is not an exact continuation. Segment stops wait until all optimizer slices of the current rollout batch are consumed. The repeated-pass sampler reconstructs its epoch-specific permutation without perturbing the global RNG. Restart tests compare an uninterrupted control with a segmented run, checking sampled outputs and parameter differences. Retention keeps the latest complete resumable checkpoint after evaluation rather than deleting all training state.

### SFT and RFT Target Construction

`sft_pilot.py` serializes the same system/user chat prefix and appends an assistant solution. It checks that tokenized full conversations share that exact prefix, masks prompt labels with `-100`, and supervises the assistant response through its chat-end marker. Overlength targets fail preparation instead of silently losing the ending. The batch collator masks padding labels too. Human-target SFT uses original MATH solutions; RFT retains correct, terminated, deduplicated model responses in sample order, with a per-problem target limit. Optional cleanup rejects excessive trailing text after the last box and unsupported characters; neither correctness filtering nor cleanup proves every reasoning step valid. Full-parameter and LoRA follow-ups remain separately configured controls.

### Evaluation and Aggregation

Greedy generation changes no parameters and differs from the sampled rollouts used during RL training. Common-stop evaluation accepts both Qwen end-of-text and chat-end tokens and records whether the completion cap was reached. A truncated response can still contain a correct final answer, so truncation and wrong-answer counts overlap. The saved outputs retain raw solutions, extracted answers and original grader decisions; later strict closed-box audits are separate reports, not silent replacements.

Sampled pass@k estimates the availability of a correct candidate, whereas consensus accuracy grades the answer actually selected. External consensus requires a strict majority of normalized, nonempty answers and otherwise uses the first candidate; no verifier or ground truth selects the output. Retrospective subset/order averages use the existing samples and are not independent runs. Aggregators reject missing or duplicate examples and mismatched prompt/configuration hashes before reporting a complete cohort. Paired fixes/breaks align the same source problems rather than subtracting unpaired accuracies.

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

`src/evaluation/problem_case_studies.py` validates saved matched development cohorts and exact quotations, exports seventeen reviewed qualitative cases, and plots outcome/length matrices. `docs/problem_case_studies/annotations.json` contains the explicit mathematical readings and evidence snippets. `tests/test_problem_case_studies.py` checks cohort guards, annotation fidelity, paired transitions, and seventeen independently authored arithmetic/symbolic checks. The results atlas includes a three-page synthesis rather than one page per case; detailed source annotations remain available without requiring a separate PDF. Full selected traces stay in scratch. This is not a trained verifier or a frequency estimate of mathematical errors.

`scripts/paper/plot_midterm.py`, `prepare_final.py`, `prepare_poster.py`, and `prepare_compendium.py` create the four document bundles. `compendium_setup.py`, `compendium_layout.py`, and `compendium_supplement.py` organize setup, numerical comparisons, and detailed diagnostics. `refresh_completed.py` only promotes complete evaluated cohorts and preserves missing checkpoint gaps. Associated `test_*report.py`, `test_compendium.py`, `test_poster.py`, and `test_completed_refresh.py` cover scope and evidence validation.

Published PDFs and referenced assets are under `docs/{midterm,final,poster,results_compendium}/`. Raw experiment data, local campaign manifests, source-linked runtime evidence, cluster job scripts, and assistant state are not required to read the documents and are not bundled as public training artifacts. The compact atlas ledger is included for its numerical records and provenance; its historical paths are references, not downloadable checkpoints.
