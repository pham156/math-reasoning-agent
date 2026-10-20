# Mathematical RLVR: 1.5B Experiment Plan

Revised proposed protocol: smaller model, more training data, four RL algorithms, no-RL controls, and a separate reward-design study. Supersedes the earlier 7B/500-problem proposal. Training code and jobs have not yet been changed to implement this plan.

## 1. Research Questions

**How do algorithm, reward-design, training-data, and exploration choices affect the performance, learning dynamics, and compute efficiency of mathematical RL post-training?**

Compare GRPO, Dr. GRPO, DAPO-style optimization, and GSPO as peer experimental conditions, using common initialization, correctness reward, data, and evaluation. No method is designated the preferred approach. Then hold the optimizer fixed to compare reward designs. Study initial-policy exploration, data selection and coverage, generalization, and reasoning behavior alongside these comparisons. Select later interventions from measured results.

Prioritize methods developed or evaluated for mathematical reasoning with verifiable rewards. These optimizers are task-general; MATH/GSM8K data, correctness rewards, and competition-math evaluation make this a mathematical reasoning study. General preference-alignment methods remain optional supporting comparisons.

## 2. Model, Updates, and Reward

| Component | Proposed choice |
|---|---|
| Common initialization | [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) **base**, pinned revision |
| Main updates | Full-parameter BF16 training, gradient checkpointing; verify memory before the grid |
| Precision control | Same BF16 weights for baseline and trained-model evaluation |
| Strong reference, optional | Frozen [Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct); report separately |
| Algorithm-study reward | R0: binary final-answer correctness; reward-study variants are defined in Section 7 |
| Answer grading | Shared final-answer extractor and pinned Math-Verify equivalence checks, with normalized exact match as a fast path |
| Correctness failures | Missing answers, parse timeouts, and malformed responses receive correctness score 0; log each reason and keep shaped reward components separate |

This changes which parameters are updated: the main study trains the whole 1.5B policy rather than the previous 7B LoRA adapters. Full training on one A100 80GB is a feasibility target, not yet a measured result. All four primary methods are critic-free. If profiling requires LoRA, freeze one common adapter configuration for every RL arm and document that protocol change before running the comparison.

Audit at least 20 reward cases covering fractions, decimals, percentages, sets, lists, LaTeX, and numeric commas. Grade the extracted final answer, not any number appearing in the reasoning. Check source-answer parseability before split creation. Reference solutions provide targets only, never demonstration traces in RL prompts. Reevaluate the baseline with the new grader; older 7B scores are historical context. The model-family reference is the [Qwen2.5-Math technical report](https://arxiv.org/abs/2409.12122); the checkpoint links above lead to the official model cards.

## 3. Datasets, Scale, and Leakage Controls

| Purpose | Source | Size and role |
|---|---|---|
| Primary training | Local MATH train, competition_math-compatible | **5,000 Level 3-5 problems**, shared by all methods |
| Development | Local MATH train | Reserve **300**, stratified by category/level, before selecting training data |
| Training coverage extension | Remaining valid MATH train, all levels | Approximately **7,200** after development reservation; exact count after grading/deduplication audit |
| Data-distribution extension | `openai/gsm8k`, `main`, train | Separate 3,000-problem easy-data condition and matched MATH/GSM8K mixture |
| Standard final benchmark | `HuggingFaceH4/MATH-500`, test | All 500; category/level breakdown; already inspected in earlier project work |
| Independent confirmation | Local MATH test | Freeze 500 excluding MATH-500 and all previously inspected test problems, subject to audit |
| Sanity/generalization | GSM8K test | Seeded 100 for pilots; all 1,319 for final models if affordable |
| Hard generalization | Local AIME 2024/2025 | 30 each; final frozen models, no checkpoint selection |

Local Arrow data contains 7,500 MATH train problems, including **5,586 at Levels 3-5**, and two with unknown levels. The 5,000 target fits after reserving development data, subject to parseability/deduplication losses; reduce all arms equally if needed. Do not use the combined MATH train/test corpus as RL training data.

Use split seed **42**, stable IDs, normalized problem hashes, and dataset snapshot hashes. Reserve a separate 300-problem GSM8K training holdout before its extension. No evaluation problem may enter screening, training, or curriculum selection. MATH-500 remains a standard comparison set; label the separate confirmation set as a custom MATH subset and verify its uninspected status before claiming that property.

## 4. Algorithms and Controls

| Arm | Method | What it tests | Method reference |
|---|---|---|---|
| B0 | **No RL**, frozen base with the training template | Matched initialization control | Official base-model card above |
| B1 | **No RL**, same base with raw problem text | Prompt-only control | Same frozen checkpoint as B0 |
| G | **GRPO** | Per-question group-standardized advantages and response-length normalization | [DeepSeekMath](https://arxiv.org/abs/2402.03300v3) |
| D | **Dr. GRPO** | Group-centered advantages and a fixed token-loss denominator | [Dr. GRPO](https://arxiv.org/abs/2503.20783v2) |
| A | **DAPO-style optimizer** | Token-level loss aggregation and asymmetric clipping for exploration | [DAPO](https://arxiv.org/abs/2503.14476v2) |
| S | **GSPO** | Length-normalized sequence importance ratios and sequence-level clipping | [GSPO](https://arxiv.org/abs/2507.18071v2) |

Use the same model, four samples/prompt, binary answer reward, uniform prompt schedule, and truncation rules in the primary comparison. Arm A isolates the DAPO optimizer components; it does not include dynamic sampling or extra length rewards and must be labeled **DAPO-style optimizer**, not the full DAPO recipe. GSPO is evaluated on our dense 1.5B model; benefits observed on larger MoE models are not assumed to transfer.

Do not relabel a trainer by its class name alone. Inspect actual advantage formulas, loss denominators, reward scaling, minibatches, and policy updates. The installed TRL GRPO configuration currently defaults to a DAPO loss, so explicitly request `loss_type="grpo", scale_rewards="group"` for G and `loss_type="dr_grpo", scale_rewards="none"` for D. Validate both with objective tests. For A, validate active-token loss normalization across the full accumulated batch, group-standardized advantages, token ratios, and the separate upper/lower clip bounds. For S, validate `exp(mean(token log-probability differences))`, sequence-level clipping, and the gradient as well as forward loss. Installed sequence-ratio options are a starting point, not proof that the selected backend exactly implements GSPO. Use objective tests and archived configuration for every method. For related methods, check advantage scaling and update rules before treating them as independent comparisons; record the precise differences if substituting a method.

Use training seeds **0, 1, 2**, keeping manifests fixed: four trained arms times three seeds = **12 main runs**. Evaluate each no-RL control once greedily and with fixed sampling seeds. Include failed/stopped runs in the comparison.

### Additional Comparisons

| Method or intervention | Priority and role |
|---|---|
| DAPO-style dynamic sampling | Next-stage math experiment: remove saturated reward groups; compare useful signal and total rollout cost |
| REINFORCE++ with group baseline | Optional reasoning comparison: global-batch advantage normalization; pin the exact variant |
| Correct-solution rejection fine-tuning | Optional mathematical non-policy-gradient baseline: train on self-generated solutions with verified answers |
| REINFORCE / PPO | Optional classical RL anchors for course analysis, after the primary methods |
| ReMax / RLOO | Lower priority: baseline-estimation comparisons rather than the main math-reasoning matrix |
| DPO / general RAFT | Lower priority: require a separate preference/filtering protocol; not primary RLVR arms |

Rejection fine-tuning must start from the same base model, use only MATH train prompts, and report both generation and SFT costs. Verify answers before accepting traces; a correct answer does not certify every reasoning step. Fix the acceptance rule, token budget, and retention of multiple successful traces before running it. Keep this baseline distinct from the frozen no-RL control and from policy-gradient RL.

## 5. Training Protocol and Compute Matching

| Setting | Initial proposed value |
|---|---|
| Policy optimizer | AdamW; LR **1e-6**, betas (0.9, 0.95), weight decay 0 |
| Schedule / gradient clipping | Constant LR; norm 1.0 |
| Training sampling | Temperature 1.0; top-p 1.0; no top-k cutoff |
| Rollouts | **4 responses per prompt**; group-size-8 extension only after profiling |
| Context budget | Prompt at most 1,024 tokens; completion at most **3,000**; total at most 4,096 |
| Duration | **One full pass over 5,000 unique problems**, rather than stopping after 50 updates |
| Logical rollout batch | 16 unique prompts, 64 responses; four minibatches of 16 responses; microbatch 1 with validated accumulation |
| Policy clipping | GRPO/Dr. GRPO: lower/upper 0.2/0.2; DAPO-style: 0.2/0.28; GSPO: initial 3e-4/4e-4, validated in pilots |
| KL reward/loss coefficient | 0; measure reference drift separately on diagnostic data |
| Checkpoints | At 0%, 25%, 50%, 75%, 100% of training-prompt exposure; also before wall-time expiry |

A full pass produces **20,000 training completions per run**, before any diagnostic sampling. There are approximately 313 logical rollout batches, but not necessarily 313 optimizer steps: four minibatches produce approximately 1,250 updates for the full pass if the backend follows this schedule. Verify and log actual counts. Freeze the prompt ordering within each seed across algorithms. Account for the final partial batch without duplicating examples to hide unequal exposure.

Use one proximal epoch and the same minibatch order for every primary method. Freeze rollout-policy log probabilities before the first minibatch, and keep complete prompt groups together so later updates expose the role of importance ratios/clipping. Report policy optimizer steps, backward token counts, clip fraction, and rollout tokens; matched prompt exposure is the primary comparison, with GPU-hour/token efficiency as a second view. If the clip fraction remains zero, do not attribute gains to clipping. Give each method the same small development-only tuning allowance if LR 1e-6 proves unstable; document tuned results separately from the fixed-setting comparison.

Clipping scales differ between token and sequence ratios; applying one numeric threshold to all methods would not create a fair comparison. Keep each method's declared thresholds and equal tuning allowance, then measure the actual update drift.

Run five-update memory tests for every backend, then a 512-problem pilot before the main runs. Target 3,000-token training from the outset. If it cannot fit, use 2,048 uniformly and label the reduced-budget protocol; do not silently truncate prompts or compare runs with different caps as equivalent. Evaluation follows the same frozen cap for all arms. Check actual tokenizer context limits before launch.

## 6. Exploration, Templates, and Data Experiments

**Before RL:** on 100 fixed development problems, compare raw text, Qwen-Math, and R1 templates. Measure answering behavior, greedy accuracy, parseability, length, and pass@8. Use the Qwen-Math template for the main training comparison and retain the raw-text control; do not mistake recovery from a poor prompt for new capability. Serialize templates explicitly; base-model raw text must bypass chat formatting.

Use eight samples at temperatures **0.4, 0.7, 1.0** for the initial exploration sweep. Qwen-Math serialization has a system instruction requesting stepwise work and a boxed final answer, followed by the problem in a user turn. R1/raw serialization is defined separately. Store exact prompts, token IDs, EOS rules, and generation settings; freeze and document each template definition without hidden instructions.

**After the algorithm study:** use the stable method to isolate one data factor at a time. Compare equal 3,000-example MATH Level 3-5, GSM8K, and 50/50 mixtures at matched prompt exposure. For coverage, compare nested 1,000/3,000/5,000 subsets at 5,000 total prompt exposures, reporting repeated versus unique problems. An all-level MATH run uses the remaining roughly 7,200 problems as a separate scaling extension.

Mixed-success selection remains a separate intervention: screen training problems with four frozen-policy samples, label 0/4, 1-3/4, 4/4, and compare matched-size random/mixed subsets with category/level balance. Report finite-sampling uncertainty, selection attrition, and screening cost. Do not cross every data/template setting with all algorithms. Repeat promising extension comparisons across seeds if affordable.

Add a bounded online dynamic-sampling experiment after the uniform optimizer study. For each generated group, retain only mixed rewards for updates, logging all-correct and all-wrong discards. Fix a total attempted-prompt budget of 5,000 and count every discarded completion toward cost; stop at that budget instead of sampling indefinitely to fill batches. Report accepted groups, unique/repeated prompts, incomplete batches, and optimizer steps. This is a bounded DAPO-style sampling adaptation, not a claim of reproducing the entire training recipe. Compare it against the same optimizer with uniform sampling at equal attempted-prompt budget and on accuracy/cost curves.

Overlong filtering is a separate loss-eligibility intervention. Format and length shaping are explicit reward-study conditions in Section 7 and remain excluded from the correctness-only optimizer matrix. Report raw answer accuracy regardless of shaped training rewards, and never label merely choosing a DAPO loss as the full recipe.

## 7. Reward-Design Experiments

**Question:** Does additional reward shaping improve answer correctness, useful reasoning, or token efficiency, or merely increase the score being optimized? This implements the reward-study direction in [the original plan](plan.md#9-rq4--reward-design).

Define `C=1` when the extracted final answer matches the reference and `C=0` otherwise, `F` as a deterministic final-answer format score, `L` as generated completion tokens excluding prompt/padding, and `T` as the fixed completion cap. Freeze these definitions, tokenization, coefficients, and failure handling before the study.

| Arm | Training reward | Purpose | Stage |
|---|---|---|---|
| R0 | `C` | Correctness-only control | Primary |
| R1 | `C + 0.10 * F` | Test whether a small format bonus improves answer completion/parseability | Primary |
| R2 | `C - 0.05 * min(L / T, 1)` | Test token efficiency and the risk of premature short answers | Primary |
| R3 | `C + 0.10 * F - 0.05 * min(L / T, 1)` | Test interaction of format and length shaping | Only after the individual comparisons |
| R4 | `C + 0.10 * (1 - C) * P_partial` | Test bounded partial credit on a trusted structured-answer subset | Grader-validation extension |
| R5 | `C + 0.10 * P_process` | Test intermediate-step feedback alongside outcome correctness | Scorer-validation extension |

Set `F=1` only when the completion contains exactly one balanced, nonempty `\boxed{...}` final answer that the shared extractor can parse, regardless of whether it is correct; otherwise `F=0`. A wrong but formatted answer can earn the 0.10 bonus, so log format-only rewarded failures explicitly. Evaluate answer correctness independently of this formatting requirement. These coefficients are proposed study settings, not claimed optimal values.

Choose one numerically stable optimizer from the algorithm study using a predeclared development-only rule: highest mean final development accuracy across seeds, with a tie broken by lower GPU cost. Freeze its loss normalization, clipping, model, prompt, dataset/order, rollout count, learning rate, and training budget for all reward arms. Every run starts from the same untouched 1.5B base checkpoint, not from an already RL-trained policy.

Run R0/R1/R2 pilots on the same 512 training problems, then the primary comparison on the same 5,000 problems with seeds 0/1/2. Reuse R0 from the algorithm study only when every setting and seed matches; otherwise rerun it. With reuse, R1/R2 add six full runs to the 12-run algorithm matrix, for **18 core training runs**. Do not cross every optimizer with every reward, and include reward-study pilot/evaluation costs in the budget.

Compare raw greedy accuracy and pass@k, format/parse failures, correct and incorrect response lengths, fixed/broken examples, reward/advantage variance, and token/GPU cost. Log `C`, `F`, length penalty, and any partial/process scores separately from total training reward. Define successful rollouts and mixed-success groups using `C`, not positive shaped rewards. Select checkpoints by raw development correctness, with token cost as a tie-breaker; shaped reward totals are not comparable measures of reasoning quality. Shorter responses count as a benefit only alongside preserved correctness, and a format bonus must not be reported as an accuracy improvement.

For R4, define `P_partial` in [0,1] only for a predeclared subset with trusted component grading, such as F1 over finite sets of roots with mathematically equivalent elements. Do not award credit for wording overlap with reference solutions. Compare R4 against a fresh R0 on exactly that subset with equal budgets; parsing failures yield zero partial credit.

For R5, define `P_process` in [0,1] using a frozen mathematical step scorer or trusted symbolic checks, validated against manually annotated held-out traces. Audit false positives, unsupported steps, and reward exploitation before training. Keep final correctness as the main reward component; the existing prompt-based verifier is not automatically a reliable process grader. Include scorer inference cost, and compare R5/R0 on the same supported subset. Process/partial rewards are explicit planned extensions, not implemented graders or certified proof verification.

Before GPU runs, extend reward tests with correct-unboxed, wrong-boxed, malformed/multiple boxes, missing answers, and paired short/long completions; check component bounds and that the same completion receives the same reward across backends. Investigate reward growth without raw-accuracy growth, format-only learning, and length reduction accompanied by lost correctness. Retain negative results rather than selecting rewards from final benchmark scores.

## 8. Evaluation Beyond Final Accuracy

| Aspect | Measurements and interpretation |
|---|---|
| Outcomes | Greedy accuracy, category/level accuracy, fixed/broken cases, paired confidence intervals |
| Exploration | Sampled pass@1/4/8, empirical success probability, all-correct/all-wrong/mixed group rates |
| Learning signal | Total reward and separate correctness/format/length/partial/process components, reward variance, advantage distribution, zero-advantage fraction |
| Optimization | Entropy, gradient norm, token/sequence clipping fractions, approximate old-policy KL, diagnostic reference KL, importance-ratio distribution |
| Sampling efficiency | Attempted/accepted groups, discard reasons, useful-gradient rate, and cost per accepted group |
| Response length | Mean/median/quantiles separately for correct, incorrect, malformed, and truncated responses |
| Format and answering | Attempts to answer versus irrelevant continuation; parse/boxing rates; repetition and truncation |
| Reasoning behavior | Reflection/backtracking frequency and audited correctness of revisions, independent of answer reward |
| Generalization | Development-to-test gap, MATH-to-GSM8K/AIME transfer, category/difficulty differences |
| Efficiency | Rollouts and tokens to a fixed accuracy target, GPU-hours, peak memory, throughput; include screening and any auxiliary-model costs |

Log per-update metrics and per-rollout records. Evaluate 100 fixed development problems at each quarter-pass checkpoint and all 300 at the end. Report the final checkpoint and the best development-selected trained checkpoint; never select using MATH-500 or AIME. Negative but numerically healthy runs still receive final evaluation. Use greedy temperature 0/top-p 1 and the frozen completion cap for every model.

For sampled evaluation, collect eight responses per problem at temperature 1/top-p 1 on a fixed 100-problem diagnostic set before/after training. Estimate pass@k as `1 - C(n-c,k)/C(n,k)` with n=8 and c correct samples; define C(a,k)=0 when a<k. Keep greedy accuracy separate from sampled pass@1. Compare fixed/broken problems against their initial sampled success; an unsuccessful eight-sample search does not establish that the original model lacked a capability.

For reflection analysis, predefine a rubric covering revisiting/checking a prior step, detecting an error, and changing a derivation. Manually audit 50 paired before/after examples per trained arm with model identity hidden. Label keyword counts as proxies; an optional judge must be validated against the audit and never supply the RL reward. Report valid versus invalid corrections. Correct final answers do not certify proof quality, and reflection frequency alone does not establish stronger reasoning.

Length analysis must also condition on level/category or compare the same problems; aggregate correct-versus-wrong length differences can reflect difficulty composition. Plot both prompt exposures and generated tokens on learning-curve axes. Report seed mean/std and paired bootstrap intervals, without treating seed variation as the same uncertainty as benchmark sampling.

## 9. Gilbreth Execution, Gates, and Artifacts

Use one A100 80GB/job, `ai-forge`, `a100-80gb`, **standby**, 64 GB host RAM, and at most four hours. Limit concurrency to two jobs. Use the isolated scratch Apptainer/venv runtime through `scripts/rl/smoke_python.sh`; this supersedes the historical `math-reasoning-grpo` environment for the 1.5B study. Code/small Slurm logs stay in home with job-ID-first names (`%j_%x` or `%A_%a_%x`). Models, data, rollouts, full checkpoints, and reports stay under the scratch paths exported by `scripts/setup_env.sh`; use `$OUTPUT_DIR/rl_1p5b_study/` for the proposed study.

Full checkpoints must restore policy, optimizer, scheduler, RNG, sampler position, counters, and any pending rollout state. Test an interrupted resume against uninterrupted training before chaining jobs. Save before expiry and continue with dependency jobs; a four-hour limit is a segment limit, not a study budget. Keep the final checkpoint plus a small rotating set of resumable checkpoints in scratch.

Do not extrapolate the old 7B/512-token LoRA timings to these runs. Measure tokens/sec, update time, and memory for each backend, then estimate `screening + sum(training + evaluation) GPU-hours`, adding reward-scorer/critic/SFT costs for selected extensions. The 12 algorithm runs generate **240,000 completions**, at most **720 million completion tokens** at the 3,000 cap. With R0 reuse, the 18 core algorithm/reward runs generate **360,000 completions**, at most **1.08 billion completion tokens**, excluding pilots/diagnostics/extensions. Set a total budget after the pilots and show the phased job matrices before submission; increasing data and length can offset the smaller model's savings.

Pause for NaN/Inf gradients or failed checkpoints. Investigate sustained missing-answer rates above 50%, sharp reference KL changes, or falling lengths together with declining raw development correctness. Length reduction alone is not a collapse criterion in R2/R3. Freeze operational thresholds from the pilots. A brief accuracy dip during template adaptation is not an automatic failure; persistent reward/format collapse requires diagnosis. Use development data for these decisions and retain stopped runs.

| Stage | Completion criterion |
|---|---|
| Setup | 1.5B snapshots, audited splits/reward, GRPO/Dr. GRPO/DAPO-style/GSPO objective tests and adapters, full-BF16 and resume smoke tests |
| Initial policy | Template/temperature exploration report and fresh controls |
| Pilot | Four backend pilots; measured memory/runtime; frozen settings and total budget |
| Main study | Four trained arms x three seeds on the same 5,000 training problems |
| Reward study | Fixed optimizer; R0/R1/R2 x three seeds; separate reward components, raw accuracy and efficiency comparisons |
| Extensions | Matched data/coverage, mixed-success or bounded dynamic-sampling experiments; optional supporting methods |
| Final report | Frozen-model benchmarks, behavior/optimization curves, fixes/breaks, cost and uncertainty |

Save configuration/environment manifests, split IDs/hashes, per-problem evaluation JSONL, per-rollout reward components/answers/lengths, checkpoints, and structured training logs. Deliver algorithm and reward-design accuracy/compute tables; reward-component/entropy/length curves; separate response-length plots for correct and incorrect answers; pass@k curves; template and data comparisons; and audited before/after reasoning examples. A controlled negative result still completes the study.

## 10. Implementation Status

Existing: 7B inference/evaluation, reward tests, subset/exploration utilities, and LoRA GRPO training. Installed source exposes GRPO/Dr. GRPO loss options and a callable-reward RLOO trainer. Required before this new protocol: 1.5B downloads, full-parameter support, explicit GRPO/Dr. GRPO/DAPO-style/GSPO objectives and validated trainer adapters, shared Math-Verify grading, R0/R1/R2 reward implementations/tests, component logging, and tested dependency/resume handling. The shared backend may expose relevant options without implementing a complete algorithm recipe; verify runtime behavior and objective gradients before launch. Partial/process rewards require separate validated graders; general-alignment baselines and rejection fine-tuning need adapters/protocols only if those extensions are selected.

Smoke implementation: `src.training.rl.smoke` and `scripts/rl/run_algorithm_smokes.slurm` provide isolated five-update, 64-problem checks of the four objectives using the downloaded 1.5B model, shared Math-Verify correctness, full-parameter mixed BF16 training, before/after inference, and checkpoint saving. The scratch Apptainer/venv stack is separate from the historical GRPO environment. CPU objective/gradient and reward tests precede GPU submission. Small smoke results do not establish benchmark improvement or exact interrupted-resume correctness; the 5,000-example/reward-design study still requires the remaining protocol gates.

Checkpointed pilot setup: all four smoke tasks passed. `src.training.rl.pilot` and `scripts/rl/submit_algorithm_pilots.sh` prepare four seed-0, 512-problem correctness-only pilots, gated by eight-update uninterrupted/restart comparisons for every method. Splits reserve 300 category/level-balanced eligible Level 3-5 development problems; the pilots use a fixed 20-problem before/after diagnostic slice to fit initial wall-time budgets. This reduced diagnostic evaluation does not replace the planned 100/300-problem evaluations. Keep tested FP32 trainable/master weights and BF16 compute uniformly across all pilots, documenting this adjustment from pure-BF16 parameters. Continuation is restricted to fully consumed rollout boundaries; interrupted mid-batch continuation is not supported. The full three-seed, 5,000-example/reward-design matrix remains deferred until restart equivalence and pilot runtime/stability are established.
