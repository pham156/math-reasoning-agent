# Midterm Research Update

Deadline supplied by the course: October 30 (week 10). This October 19, 2026 draft focuses on GRPO and Dr. GRPO, full-parameter RL fine-tuning, an initial SFT control, exploration, external benchmarks, paired-seed learning-rate tuning, and KL/clipping ablations. Dr. GRPO is selected for report scope because its MATH-500 score (65.2%) is highest among the screened non-GRPO variants; this is a retrospective presentation choice, not an independent method-ranking test. The broader project experiments and plan are unchanged.

## Overleaf

Upload `midterm_overleaf.zip` as a new Overleaf project, set `main.tex` as the main document, and select pdfLaTeX. The source uses the standard `IEEEtran` conference class with its default two-column layout and font size. References are embedded, and the ZIP includes the vector PDF under `figures/`; no BibTeX run or separate image upload is required. Replace `Project Team` in the author block with the actual author names; no names or email addresses have been inferred.

The intended length is two pages including references. Do not change margins or shrink the body font to force the limit; edit the prose if author details or new results change pagination.

The Index Terms block is omitted. `Future Work` stays within the experimental plan: more updates and training data, additional algorithms, further RL/SFT tuning, correctness/format/length reward comparisons, and controlled hyperparameter/data ablations. Lean, formal proofs, and broader aspirational directions are excluded. The experimental design emphasizes model, data, correctness reward, method comparison, and evaluation without GPU/scheduler/precision details or incidental numeric settings. One merged fine-tuning table compares no training, SFT, GRPO, and Dr. GRPO across development and external benchmarks; SFT external entries are dashes because they were not evaluated. The numerical results remain in the table/body rather than the qualitative abstract. Expanded SFT and learning-dynamics discussion fills the two-page report without changing its font size or margins. These report edits do not alter the operational experiment plan or completed results.

The source uses `IEEEtran` in conference mode. Its standard abstract appears at the top of the first (left) column below the full-width title/author block, not as a full-width abstract. This differs from other IEEE formats; the report retains the conference class's native placement rather than manually spanning columns.

The PDF is checked for two-page pagination, typography, complete references, and readable figure panels after each edit. Overleaf/pdfLaTeX itself is not executed locally; the local build uses Tectonic's XeTeX engine with explicit T1 font encoding to preserve IEEE Times typography.

Validation covers two-page pagination, TeX warnings, visual inspection, plot-helper tests, and Ruff. All plotted source observations are exported; pass@k and the learning-rate results retained in the text must match their validated summaries. Entropy requires finite measurements at all 128 updates per method. The combined development/external table and ZIP contents (LaTeX plus vector figure) are checked against the final source artifacts.

The entropy revision exports 344 source observations for the selected methods only. Six plot-helper tests, Ruff, and targeted diff checks passed; the final PDF has exactly two pages and no TeX warnings. Both pages were visually inspected, entropy raw values/smoothing were checked against training logs, and the Overleaf source/vector figure match the final files.

The report keeps the GRPO/Dr. GRPO comparison and external benchmarks at the original seed0/LR1e-6, with an untouched no-RL control. The SFT control has common-stop dev accuracy 40.3%, uses reference-solution targets, and is data-matched rather than compute-matched. It has no external benchmark or sampled-coverage scores in this report. Only the GRPO LR comparison has two seeds. KL/clipping variants use seed0/LR1e-6 and must not be presented as tested combinations with LR3e-6. No higher-LR external benchmark scores are claimed. Fixed-prefix diagnostic KL and sampled training KL are different statistics.

## Local Build

From the project root, run `bash scripts/paper/build_midterm.sh`. It uses the existing scratch Apptainer helper image and a pinned Tectonic 0.15.0 binary. Tool downloads, TeX bundles/cache, and intermediate files stay in scratch; only the final PDF and Overleaf ZIP are exported to this home directory. No host system packages or training environment dependencies are changed.

## Line Plots

The paper includes a full-width three-panel vector figure: (a) GRPO/Dr. GRPO seed0/LR1e-6 training correctness reward, (b) these two models and the no-RL control's sampled pass@k, and (c) training-policy token entropy for the same two runs. RL method colors/styles are consistent across all panels. The frozen baseline appears only in(b), explicitly labeled `No RL (b only)`; it has no training trajectory. DAPO-style and GSPO are omitted from the focused figure and results table. Learning-rate results remain in the text, not panel(c).

Panel(a) shows a trailing four-observation mean over 32 recorded rollout measurements/method at updates 1,5,...,125; four observations cover up to 16 updates. Panel(b) computes integer k=1,...,8 directly from existing eight responses on the same 100 problems; k=1,4,8 must match the saved exploration summary. Panel(c) shows a trailing eight-update mean of trainer-logged token entropy, with 128 raw measurements per method. These are training-context measurements, not a fixed-problem entropy evaluation, certified reasoning quality, or a direct estimate of solution coverage. No artificial error bands or unmeasured cross-seed uncertainty are added. pass@k is finite-sample oracle coverage, not answer selection.

Reproduce the plots with `RL_SMOKE_PYTHON=$SCRATCH_ROOT/env/rl-plots/bin/python bash scripts/rl/smoke_python.sh -m scripts.paper.plot_midterm`, then rebuild the paper. `figures/rl_dynamics_and_exploration.pdf` is the embedded vector figure, `.png` its preview, and `figures/plotted_data.csv` contains 344 raw/plotted observations: 64 training-reward observations, 24 pass@k estimates, and 256 entropy measurements. Plotting validates completeness, held-out IDs, LR settings, and summary agreement. All figure artifacts live in home; Matplotlib cache remains scratch. The entropy panel replaces the two-point learning-rate panel.

## Evidence And Scope

- Design: `docs/rl_experiment_plan.md`, plus the implemented settings in `src/training/rl/{smoke,pilot,methods}.py`. The paper follows the actual FP32-parameter/BF16-compute pilots, not superseded pure-BF16 proposals.
- Greedy, stopping controls, and sampled exploration: `outputs/rl_1p5b_study/controls_exploration_20261014/summary.{md,json}`.
- Learning rates: `outputs/rl_1p5b_study/lr_tuning_20261014/summary.{md,json}`.
- Upward GRPO rates: `outputs/rl_1p5b_study/grpo_lr_upper_20261016/summary.{md,json}`. Dev300:1e-6=72.3%,2e-6=70.0%,3e-6=75.0%;3e-6 fixes21/breaks13 relative to1e-6, checked from matching per-problem JSONL. Higher-rate checkpoints have no external benchmark scores.
- Paired-seed LR and KL/clipping controls: `outputs/rl_1p5b_study/grpo_controls_20261017/summary.{md,json}`. LR1e-6 versus3e-6 is72.3/75.0% for seed0 and70.3/73.3% for seed1; two-seed means71.33/74.17%. KL.001/.01 yields70.3/71.0%; clip.1/.3 yields68.7/70.7%, all against original72.3% seed0/LR1e-6. Fixed-prefix KL uses eight shared baseline-response contexts, not an on-policy or sequence-level estimate. All four KL/clipping restart gates passed exact rollout/weight equivalence. Summary aggregation was executed directly on CPU while its Slurm allocation waited on priority.
- Retention: `outputs/rl_1p5b_study/retention_20261014/summary.{md,json}`.
- Completed external benchmarks: `outputs/rl_1p5b_study/external_benchmarks_20261015/summary.{md,json}` and `submission.json`. Original1e-6 models only. All500 MATH-500 test problems and seed42 random100 GSM8K test problems, common greedy3000-token protocol, no retrieval/verifier. Normalized-text overlap with512 training problems was zero; unknown pretraining contamination and historical benchmark inspection remain limitations.
- Lean failure: `outputs/rl_1p5b_study/lean_smoke_20261014/checker_preflight.json`; no formal-proof success is claimed.

Training uses 512 MATH-train Level 3-5 problems; development uses 300 disjoint problems from that same source split. The paper does not relabel development scores as MATH-500 results, combine historical 7B scores with current 1.5B scores, claim multi-seed significance, or describe the DAPO-style optimizer as the complete DAPO recipe. Correct final answers are not claimed to establish correct proofs.

Primary references were checked against their arXiv records: [Qwen2.5-Math](https://arxiv.org/abs/2409.12122), [MATH](https://arxiv.org/abs/2103.03874), [GSM8K](https://arxiv.org/abs/2110.14168), [GRPO](https://arxiv.org/abs/2402.03300), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [DAPO](https://arxiv.org/abs/2503.14476), and [GSPO](https://arxiv.org/abs/2507.18071). Formatting follows the [IEEE conference template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn).
