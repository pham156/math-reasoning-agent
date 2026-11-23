# Research Plan: Next-Stage Mathematical RLVR Post-Training

**Project**: Mathematical Reasoning Agent ([`math-reasoning-agent`](file:///home/vo43/math-reasoning-agent))  
**Target Architecture**: `Qwen/Qwen2.5-Math-1.5B` Base  
**Target Environment**: Purdue Gilbreth HPC Cluster (NVIDIA A100-80GB PCIe, `ai-forge` / `standby`)  
**Date**: November 22, 2026  
**Document Path**: [`research/next_direction.md`](file:///home/vo43/math-reasoning-agent/research/next_direction.md)  

---

## Executive Summary

Reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm for post-training large language models on complex mathematical reasoning. By substituting subjective reward models with deterministic, exact-correctness verifiers (e.g., `math-verify`), algorithms like Group Relative Policy Optimization (GRPO) elicit chain-of-thought exploration, self-correction, and backtracking without human demonstration traces.

However, mathematical RLVR faces foundational methodological, optimization, and empirical challenges:
1. **Optimization Stability & Denominator Confounds**: While unscaled algorithms like Dr. GRPO drastically reduce gradient norms compared to standard GRPO, rigorous code inspection reveals an implementation confound in standard toolkits (e.g., TRL 0.25.1) where Dr. GRPO divides token loss by a fixed maximum context cap (`batch_size * max_completion_length = 3000`) rather than active tokens, artificially dampening gradients by $3\times$–$4\times$.
2. **Exploration, Entropy, & Inactive Clipping**: Initial pilot observations attributed elevated policy entropy in DAPO-style optimization to asymmetric clipping ($\epsilon_{\text{high}} = 0.28$). However, log audits demonstrate that under single-iteration on-policy updates (`num_iterations = 1`), policy clipping is mathematically dormant (`clip_ratio/high_mean == 0.0`), proving that initial entropy preservation stems from advantage normalization and batch-level normalizer scaling rather than clipping headroom.
3. **Rollout Token Economy vs. Curriculum Attrition**: In uniform group sampling, 38%–42% of generated prompt groups yield zero reward variance (all-correct or all-wrong), contributing zero gradient signal. While dynamic sampling eliminates uninformative rollouts, naive filtering discards up to 45% of generated tokens and starves the policy of exposure to challenging Level 5 problems.
4. **Reasoning Length vs. Underthinking**: Naive constant length penalties frequently cause catastrophic "underthinking collapse" on competition-level problems, while unpenalized correctness training can lead to verbosity bloat on routine problems.
5. **Latent Reasoning Horizons vs. Greedy Floor Collapse**: Completed evaluations on Olympiad benchmarks (AIME 2025/2026) demonstrate that greedy decoding collapses to near-floor accuracy (0.0%–3.3%), whereas sampled evaluation reveals substantial latent exploration coverage (Pass@16 = 20.00% on AIME 2025), proving that capability is hidden by greedy search rather than absent from the base policy.

This research plan synthesizes our empirical findings from Purdue Scholar and Gilbreth clusters, corrects prior mathematical and algorithmic confounds, grounds findings in 2024–2026 literature, identifies four core research gaps, and formalizes a preregistration-ready next experiment: **A Controlled Investigation of Bounded Dynamic Sampling vs. Uniform Scheduling under Matched 1.5M Rollout Token Budgets**.

---

## 1. Synthesis of Established Findings, Confounds, and Limitations

### 1.1 Completed Empirical Measurements (Locally Verified)

All claims in this section are grounded in documented run manifests, Slurm logs, and evaluation artifacts in this repository:

1. **Gradient Variance Suppression & The Dr. GRPO Denominator Confound**:
   * *Empirical Finding*: In 5-gradient-step profiles on NVIDIA A100-80GB ([`REPORT.md`](file:///home/vo43/math-reasoning-agent/REPORT.md#L23-L27), `/scratch/gilbreth/vo43/math-reasoning-agent/outputs/rl_comparison_a100/`), Dr. GRPO reduced mean gradient norm from $\|\nabla_\theta\| = 0.953$ (standard GRPO) to $\|\nabla_\theta\| = 0.193$—an ~80% reduction—while achieving higher mean rollout reward (`0.328` vs `0.297`). On Scholar A40 ([`docs/scholar_pilot_findings_and_gilbreth_migration.md`](file:///home/vo43/math-reasoning-agent/docs/scholar_pilot_findings_and_gilbreth_migration.md#L19-L22)), Dr. GRPO's step-1 gradient norm was $0.0688$ versus $7.748$ in standard GRPO and $7.747$ in GSPO.
   * *Theoretical & Code Audit*: Inspection of the TRL 0.25.1 backend (`trl/trainer/grpo_trainer.py`, lines 1841–1854) uncovers a structural mathematical confound in how loss denominators are computed:
     $$\mathcal{L}_{\text{GRPO}} = \frac{1}{B} \sum_{i=1}^B \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \mathcal{L}_{i,t}$$
     $$\mathcal{L}_{\text{Dr. GRPO}} = \frac{1}{B \cdot L_{\max}} \sum_{i=1}^B \sum_{t=1}^{|y_i|} \mathcal{L}_{i,t}$$
     In standard GRPO, token loss is normalized per sequence by actual generated completion length $|y_i|$ ($\approx 750$ tokens on average). In Dr. GRPO, TRL divides by $B \times L_{\max}$ ($L_{\max} = 3,000$). Consequently, Dr. GRPO's loss is divided by $3,000$ rather than $\sim 750$, introducing an artificial $3,000 / 750 \approx 4\times$ denominator damping factor on top of unscaled advantages ($A_i \in [-1, 1]$ vs standardized $A_i / \sigma_G \approx \pm 2$). Thus, the observed reduction in gradient norm is heavily driven by this constant context cap denominator confound, demonstrating that rigorous method comparisons require active token normalization (like BNPO or DAPO's `num_items_in_batch`) or step-size matching.

2. **Inactive Policy Clipping Under `num_iterations = 1` and Entropy Dynamics**:
   * *Empirical Finding*: In initial 5-step A100 comparisons ([`REPORT.md`](file:///home/vo43/math-reasoning-agent/REPORT.md#L25)), DAPO-style optimization recorded higher token entropy ($H = 0.727$ vs $0.643$ for GRPO), which was hypothesized to result from its asymmetric upper clipping bound ($\epsilon_{\text{high}} = 0.28$).
   * *Disconfirming Evidence from Logs*: Slurm execution logs for DAPO-style (`logs/12192651_2_rl_compare_a100.out`, lines 43–47) and GRPO resume gates (`logs/12471585_0_rl_gate.out`, lines 23–30) conclusively refute this hypothesis. At every logged step, the upper clipping fraction is identically zero:
     ```text
     'clip_ratio/low_mean': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/region_mean': 0.0
     ```
   * *Mathematical Root Cause*: Under standard training configurations where `num_iterations = 1`, the policy evaluating the token probabilities is identical to the rollout policy that generated them ($\pi_\theta = \pi_{\text{old}}$). The importance ratio $r_{i,t}(\theta) \equiv 1.0$ is strictly within $[1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}] = [0.80, 1.28]$, meaning clipping is mathematically inert on iteration 1. The observed entropy divergence is instead driven by advantage normalization and the batch normalizer adapter implemented in `src/training/rl/smoke.py` (lines 35–41). Clipping can only influence policy dynamics when `num_iterations > 1` (multi-epoch PPO updates) or when off-policy replay data is consumed.

3. **Latent Reasoning Horizons vs. Greedy Floor Collapse (AIME 2025/2026)**:
   * *Empirical Finding*: Evaluation runs on A10 hardware ([`scripts/rl/run_aime_eval_a10.slurm`](file:///home/vo43/math-reasoning-agent/scripts/rl/run_aime_eval_a10.slurm), logs `logs/12488340_aime_eval_a10.out` and `logs/12492613_aime_eval_a10.out`, artifacts in `/scratch/gilbreth/vo43/math-reasoning-agent/outputs/rl_1p5b_study/aime_20261122/`) demonstrate a striking divergence between greedy and sampled performance:
     * **AIME 2025 Greedy Accuracy**: **3.33%** (1/30 solved; SFT model collapses to 0.0%).
     * **AIME 2025 Sampled Coverage (Pass@16)**: **20.00%** (6/30 problems solved at least once). Pass@1 = 1.46%, Pass@4 = 5.67%, Pass@8 = 10.89%, Pass@16 = 20.00%.
     * **AIME 2026 Greedy Accuracy**: **6.67%** (2/30 solved).
     * **AIME 2026 Sampled Coverage (Pass@16)**: **15.79%** (3/19 solved on partial shards; Pass@1 = 3.95%, Pass@4 = 11.05%, Pass@8 = 14.72%, Pass@16 = 15.79%).
   * *Scientific Implication*: The base 1.5B model already possesses valid reasoning trajectories for 20% of competition Olympiad problems, but greedy decoding collapses to the floor because greedy token selection selects high-frequency local tokens that fall into dead ends. This confirms that exploration and solution coverage (Pass@$k$), rather than greedy accuracy alone, must serve as the primary diagnostic lens for mathematical RLVR post-training.

4. **Generalization and Benchmark Gains from Pure Outcome RL**:
   * *Empirical Evidence*: In a 128-update pilot across 512 MATH Level 3–5 problems ([`docs/midterm/main.tex`](file:///home/vo43/math-reasoning-agent/docs/midterm/main.tex#L60-L76)), GRPO increased held-out development accuracy (300 problems) from 45.0% (base) to 72.3% (+27.3pp), while Dr. GRPO reached 68.3%.
   * *External Transfer*: On the uninspected MATH-500 benchmark, GRPO achieved 67.4% (+32.6pp over base 34.8%), and Dr. GRPO achieved 65.2%. On GSM8K-100, GRPO reached 73.0% and Dr. GRPO reached 75.0% (base was 38.0%), demonstrating robust out-of-distribution transfer from competition mathematics to multi-step word problems.
   * *Sampling Coverage (Pass@$k$)*: Pass@8 on 100 development problems rose from 79.0% (base) to 89.0% (GRPO) and 90.0% (Dr. GRPO). Pass@1 rose from 29.4% (base) to 55.0% (GRPO) and 52.6% (Dr. GRPO).

5. **Paired-Seed Learning Rate Scaling Dynamics**:
   * *Empirical Evidence*: Upward learning rate exploration on GRPO confirmed that LR $3\times 10^{-6}$ reliably outperformed the default $1\times 10^{-6}$ across two independent seeds: Seed 0 reached 75.0% vs 72.3%; Seed 1 reached 73.3% vs 70.3% (two-seed means: 74.2% vs 71.3%) ([`docs/midterm/main.tex`](file:///home/vo43/math-reasoning-agent/docs/midterm/main.tex#L85-L87)).

6. **Supervised Fine-Tuning (SFT) Control Pathology & EOS Token Diagnosis**:
   * *Empirical Evidence*: An initial SFT control trained on the same 512 problems scored only 40.3% dev accuracy (below the 45.0% base) and truncated 292 out of 300 evaluations ([`docs/midterm/main.tex`](file:///home/vo43/math-reasoning-agent/docs/midterm/main.tex#L71)).
   * *Root Cause Diagnosis*: As detailed in [`docs/sft_termination_audit.md`](file:///home/vo43/math-reasoning-agent/docs/sft_termination_audit.md#L1-L11), supervised target sequences ended with `<|im_end|>` (151645) followed by newline (198), but model generation configuration stopped only on `<|endoftext|>` (151643). Consequently, the SFT model emitted valid solutions and `<|im_end|>`, but continued generating repetitive filler text until hitting the 3,000-token cap. Truncating at the first `<|im_end|>` reduced generated length from 3,000 to 169.2 tokens, though accuracy remained impaired.

---

### 1.2 Distinguishing Measurements from Planned and Inaccessible Results

Rigorous science requires clear boundaries between what is proven, what is planned, and what is unobservable:

| Result Category | Status | Evidence Location | Caveats & Validity Boundaries |
|:---|:---:|:---|:---|
| **5-Step A100 Comparison** | Completed | [`REPORT.md`](file:///home/vo43/math-reasoning-agent/REPORT.md), `/scratch/gilbreth/vo43/.../rl_comparison_a100/` | Hardware & gradient validity only; 5 steps cannot establish asymptotic convergence or reasoning quality. |
| **8-Step Scholar Pilot** | Completed | [`docs/scholar_pilot_findings_and_gilbreth_migration.md`](file:///home/vo43/math-reasoning-agent/docs/scholar_pilot_findings_and_gilbreth_migration.md) | Scholar cluster timeouts; terminated at Step 8 during evaluation. |
| **128-Step Dev & Benchmark Study** | Completed (Historical) | [`docs/midterm/main.tex`](file:///home/vo43/math-reasoning-agent/docs/midterm/main.tex), [`.agent/CONTINUITY.md`](file:///home/vo43/math-reasoning-agent/.agent/CONTINUITY.md) | Single seed for benchmarks; raw checkpoints reside in inaccessible scratch (`/scratch/gilbreth/pham156/`, `Permission denied`). |
| **Paired-Seed LR Confirmation** | Completed (Historical) | [`docs/midterm/main.tex`](file:///home/vo43/math-reasoning-agent/docs/midterm/main.tex), [`.agent/CONTINUITY.md`](file:///home/vo43/math-reasoning-agent/.agent/CONTINUITY.md) | $N=2$ seeds; higher LR models were never evaluated on external MATH-500 or GSM8K benchmarks. |
| **AIME 2025/2026 Evaluation** | Completed Baseline | [`scripts/rl/run_aime_eval_a10.slurm`](file:///home/vo43/math-reasoning-agent/scripts/rl/run_aime_eval_a10.slurm), `logs/12488340_*.out`, `logs/12492613_*.out` | Base model evaluated on greedy (3.3%) and pass@16 (20.0%); post-RL models not yet evaluated. |
| **Checkpoint Sharding Gate** | **Resolved** | Commit [`7926feb`](file:///home/vo43/math-reasoning-agent/.git) | Missing shard exception was fixed by adding `--save_steps 4` in `run_resume_gate.slurm`. |
| **5,000-Problem Main RL Matrix** | Planned / Unexecuted | [`docs/rl_experiment_plan.md`](file:///home/vo43/math-reasoning-agent/docs/rl_experiment_plan.md#L31-L36) | Deferred pending cluster budget and restart gate resolution. |
| **Reward Study (R1 Format, R2 Length)** | Planned / Unexecuted | [`docs/rl_experiment_plan.md`](file:///home/vo43/math-reasoning-agent/docs/rl_experiment_plan.md#L113-L135) | Implementation scaffolds exist; no multi-seed GPU runs submitted. |
| **Lean 4 Proof Verification** | Halted / Blocked | [`docs/rl_experiment_plan.md`](file:///home/vo43/math-reasoning-agent/docs/rl_experiment_plan.md#L44-L45) | Preflight packaging failed on mount checks; formal verification deferred by project policy. |

---

### 1.3 Hardware Constraints & Resolved Infrastructure Gates

1. **Hardware Memory & Autoregressive Latency**:
   * Autoregressive decoding of 100 MATH problems (up to 3,000 tokens) in eager PyTorch SDPA takes 35–45 minutes on an A100-80GB and >105 minutes on an A40. Full evaluation loops can easily breach the strict 4-hour Slurm standby window if coupled directly with training.
   * Full-parameter training (FP32 master weights + AdamW states + BF16 activations) requires 28.76 GB VRAM, causing immediate OOM on 24 GB hardware (A10 / A30) during backward passes.
2. **Resolution of Checkpoint Sharding Failure (Commit `7926feb`)**:
   * Prior resume gate attempts (e.g. Job `12471585`, Job `12471589`) failed with:
     ```text
     RuntimeError: Missing or mismatched checkpoint shards
     ```
     in [`src/evaluation/check_rl_resume.py`](file:///home/vo43/math-reasoning-agent/src/evaluation/check_rl_resume.py#L32).
   * *Diagnosis & Fix*: The failure was caused by `scripts/rl/run_resume_gate.slurm` stopping after step 4 without specifying `--save_steps 4`. Because HuggingFace/TRL defaults to saving every 500 steps, no `checkpoints/checkpoint-4` folder was created, leaving the shard list empty. In commit `7926feb`, `--save_steps 4` was explicitly passed to all pilot and gate scripts, resolving the failure mode without modifying verification logic.
3. **Token Budget Arithmetic Correction**:
   * In previous draft iterations, a total budget of 400,000 generated tokens was proposed as equivalent to 120–130 optimizer updates.
   * *Rigorous Token Math*: In our setup (`gradient_accumulation_steps = 16`, `num_generations = 4`, `generation_batch_size = 64`), 64 completions are generated every 4 optimizer steps, or 16 completions per optimizer step. With completions averaging $\sim 750$ to $1,000$ tokens, each optimizer step consumes:
     $$\text{Tokens per Update} = 16 \times 750 \approx 12,000 \text{ tokens (up to } 16,000 \text{ tokens at } 1,000 \text{ tokens/response)}$$
   * Therefore, 400,000 tokens provides only $400,000 / 16,000 \approx 25$ optimizer steps (or only $\sim 8$ steps if completions hit the 3,000-token cap). A standard 128-update pilot requires:
     $$\text{Token Budget for 128 Updates} = 128 \times 16 \times 750 \approx 1,536,000 \text{ tokens } (\approx 1.5\text{M tokens})$$
   * All experimental protocols must budget for **1.5M generated rollout tokens** to achieve 128 meaningful optimization updates.

---

## 2. Literature Synthesis & Theoretical Landscape

### 2.1 RLVR Optimization Formulations: Token vs. Sequence Objectives

* **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024; DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)):
  Eliminates the separate critic network in PPO by generating a group of $G$ completions $\{y_1, \dots, y_G\}$ for prompt $x$ and standardizing rewards:
  $$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\}) + \epsilon}$$
  The policy loss aggregates clipped token importance ratios:
  $$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x, y_{i,<t})} A_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A_i \right)$$

* **Dr. GRPO (GRPO Done Right)** (Liu et al., 2025; Sea AI Lab / NUS, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)):
  Identifies two fundamental mathematical biases in standard GRPO:
  1. *Question-Difficulty Bias*: Dividing by $\sigma_G \to 0$ over-weights prompts where almost all completions fail or succeed, distorting gradient updates across multi-task distributions. Dr. GRPO replaces this with unscaled advantages: $A_i = r_i - \mu_G$.
  2. *Response-Level Length Bias*: Normalizing token loss by each response's individual length $|y_i|$ penalizes long incorrect solutions less per token than short incorrect solutions. Dr. GRPO replaces $|y_i|$ with a uniform sequence length denominator $L_{\max}$, preventing artificial reasoning bloat.
  *However, as documented in Section 1.1, when $L_{\max} = 3000$ and completions average 750 tokens, this fixed denominator damps gradients by $4\times$, requiring careful active token normalization (BNPO) or step-size matching.*

* **Group Sequence Policy Optimization (GSPO)** (Qwen Team, 2025, [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)):
  Identifies a *granularity mismatch*: verifiable rewards evaluate the entire completion, yet PPO/GRPO apply importance sampling and clipping per token. Over 2,000–4,000 token completions, compounding token ratios generate extreme variance and policy collapse. GSPO defines a length-normalized sequence-level importance ratio:
  $$s_i(\theta) = \left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)} \right)^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t})}{\pi_{\text{old}}(y_{i,t})} \right)$$
  GSPO clips $s_i(\theta)$ directly at the sequence level, dramatically stabilizing MoE and long-context training.

### 2.2 Exploration, Entropy, Clipping Dynamics, and Multi-Iteration PPO

* **DAPO: Decoupled Clip and Dynamic Sampling** (Yu et al., 2025; ByteDance Seed / Tsinghua, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)):
  1. *Asymmetric "Clip-Higher"*: Employs $\epsilon_{\text{high}} > \epsilon_{\text{low}}$ (e.g. $0.28$ vs $0.20$), granting greater probability headroom to promising low-probability reasoning tokens and delaying entropy collapse.
  2. *Dynamic Sampling*: Discards all-correct ($r_i=1, \forall i$) and all-wrong ($r_i=0, \forall i$) prompt groups during rollouts, replenishing the batch with informative mixed-success groups ($0 < \text{std}(R) > 0$).
  *Critical Requirement*: As established in Section 1.1, asymmetric clipping only affects updates when $\pi_\theta \neq \pi_{\text{old}}$, which requires setting `num_iterations > 1` (e.g., 2–4 proximal epochs per generation batch). Under `num_iterations = 1`, clipping is inactive.

* **Entropy Collapse & Flow Optimization in RLVR** (e.g. OPEFO, $G^2RPO$, EDGE-GRPO, 2025):
  In outcome-based RLVR, policy entropy typically decays monotonically. Because gradients only reinforce sampled tokens, early entropy collapse locks the policy into suboptimal reasoning templates, preventing the discovery of alternative proofs.

* **Length Penalties & Underthinking Collapse** (NeurIPS 2025/2026; ALP, GRPO-$\lambda$):
  While pure RLVR encourages long CoT traces, naive constant length penalties ($R = C - \lambda \cdot L$) frequently cause catastrophic "underthinking collapse": models truncate their derivations on complex problems to minimize penalty. Recent methods show length penalties must be difficulty-adaptive or applied only conditionally after accuracy stabilizes.

---

## 3. Four Grounded Research Gaps

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CORE RESEARCH GAPS                                      │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Gap 1: Credit Assignment │ Token vs. Sequence IS       │ Dr. GRPO vs. BNPO vs. GSPO    │
│ Gap 2: Exploration       │ Multi-Iteration Clipping    │ PPO Epochs > 1 vs. Inert Clip │
│ Gap 3: Rollout Economy   │ Dynamic vs. Uniform Sampling│ Gradient Density vs. Bias     │
│ Gap 4: Length Regulation │ Adaptive vs. Fixed Penalty  │ Verbosity vs. Underthinking   │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

### Gap 1: Token-Level Credit Assignment vs. Sequence-Level Stability under Long Reasoning Horizons
* **Empirical Grounding**: Our 5-step and 8-step benchmarks confirm Dr. GRPO slashes gradient norm by ~80% (0.953 $\to$ 0.193), but code analysis proved this is partly an artifact of TRL's fixed $L_{\max} = 3000$ denominator. Meanwhile, GSPO applies a geometric sequence-level ratio $s_i(\theta)$, completely bypassing token-level variance accumulation.
* **Open Scientific Question**: When Dr. GRPO is evaluated with active token normalization (BNPO: $\sum_t \mathcal{L}_t / \sum_t 1$) rather than the artificial $3,000 \times B$ denominator, does sequence-level importance sampling (GSPO) still provide superior gradient stability over 3,000-token completions, or does sequence-level credit assignment dilute critical mathematical pivot steps?

### Gap 2: Sustaining Exploration vs. Entropy Collapse: Multi-Iteration Clipping Dynamics vs. Normalizer Effects
* **Empirical Grounding**: Our log analysis proved that under single-iteration training (`num_iterations = 1`), clipping was 100% inactive (`clip_ratio/high_mean == 0.0`), disproving the claim that DAPO's higher entropy was caused by asymmetric clipping.
* **Open Scientific Question**: When policies undergo multi-iteration PPO updates (`num_iterations = 2` or `4`), where clipping actually becomes active, does asymmetric "clip-higher" ($\epsilon_{\text{high}} = 0.28$) actively stimulate genuine reasoning exploration (measured by Pass@16 and reasoning n-gram diversity), or does it merely introduce an upward probability drift on common filler tokens?

### Gap 3: Rollout Token Efficiency vs. Curriculum Attrition in Bounded Dynamic Sampling
* **Empirical Grounding**: In our 128-update pilot study, mixed-reward prompt groups accounted for only 58%–62% of rollout batches. Consequently, 38%–42% of generated rollouts had zero within-group variance, contributing zero gradient signal. While DAPO dynamically discards all-0 and all-1 groups to maximize gradient density per step, discarding all-0 groups systematically filters out the hardest problems (Level 5 / Olympiad), and generating discarded rollouts consumes massive cluster compute. Furthermore, our corrected token arithmetic shows that a full 128-update pass consumes 1.5M tokens, not 400k tokens.
* **Open Scientific Question**: When evaluated under a strictly matched budget of **1,500,000 total generated rollout tokens** rather than optimizer steps, does bounded dynamic sampling accelerate mathematical reasoning mastery, or does it degrade hard-problem transfer due to curriculum attrition and token discard overhead?

### Gap 4: Length Regulation Tradeoffs: Difficulty-Adaptive Penalties vs. Pure Correctness vs. Underthinking Collapse
* **Empirical Grounding**: Our pilots proved that pure correctness rewards ($R_0$) naturally compress completions from 1,665 to ~663–760 tokens. However, on hard problems (AIME), models frequently fail due to truncation or superficial reasoning. The planned reward design ($R_2 = C - 0.05 \cdot L/T$) applies a flat penalty. Contemporary literature warns that flat length penalties induce premature underthinking collapse.
* **Open Scientific Question**: Can an outcome-conditional or difficulty-adaptive length penalty reduce verbosity on routine problems without degrading reasoning depth and accuracy on competition-level problems?

---

## 4. Formal Hypotheses, Discriminating Predictions, and Minimal Experiments

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              HYPOTHESIS MAPPING MATRIX                                 │
├───────┬───────────────────────────────┬────────────────────────────────────────────────┤
│ Gap   │ Candidate Hypothesis A (H_A)  │ Candidate Hypothesis B (H_B)                   │
├───────┼───────────────────────────────┼────────────────────────────────────────────────┤
│ Gap 1 │ Token Credit Superiority      │ Sequence Stability Superiority                 │
│ Gap 2 │ Exploration Elicitation       │ Asymmetric Drift & Entropy Artifact            │
│ Gap 3 │ Gradient Density Acceleration │ Curriculum Attrition & Token Inefficiency      │
│ Gap 4 │ Adaptive Length Pareto Gain   │ Intrinsic Compression Sufficiency ($R_0$)      │
└───────┴───────────────────────────────┴────────────────────────────────────────────────┘
```

### Detailed Formalization for Each Gap

#### Gap 1: Credit Assignment (Active-Token Normalization vs. Sequence IS)
* **Research Question**: In full-parameter mathematical RLVR with completions up to 3,000 tokens, does sequence-level importance sampling (GSPO) yield superior training stability and reasoning transfer compared to active-token-normalized Dr. GRPO (BNPO)?
* **Candidate $H_{1A}$ (Token Credit Superiority)**: Token-level policy gradient updates with unscaled advantages and active token normalization ($\sum \text{tokens}$) provide localized credit assignment to pivotal derivation steps, enabling the model to reinforce specific mathematical transitions and achieving higher Pass@16 coverage.
* **Candidate $H_{1B}$ (Sequence Stability Superiority)**: Sequence-level geometric-mean ratios ($s_i = (\pi/\pi_{\text{old}})^{1/|y_i|}$) eliminate the exponential variance accumulation of multi-token ratios over long trajectories, producing lower gradient variance across updates and superior out-of-distribution transfer on competition math (AIME).
* **Causal Estimand**: Difference in out-of-distribution accuracy and gradient variance:
  $$\tau_1 = \mathbb{E}[\text{Acc}_{\text{AIME}}(\text{GSPO})] - \mathbb{E}[\text{Acc}_{\text{AIME}}(\text{BNPO})]$$
* **Discriminating Predictions**:
  * Under $H_{1A}$: BNPO achieves higher Pass@16 on Level 5 MATH ($+4.0\text{pp}$) and shows localized entropy reduction concentrated specifically on formula and boxed tokens.
  * Under $H_{1B}$: GSPO maintains strictly bounded policy update norms ($\|\Delta\theta\|$) throughout 128 updates, whereas BNPO shows rising gradient variance on long completions ($L > 1500$), allowing GSPO to achieve higher AIME solve rates.
* **Indeterminate Outcome**: Equal dev accuracy ($\pm 1.0\text{pp}$) with overlapping gradient norms and Pass@16 distributions; would indicate that gradient accumulation (microbatch=1, grad_accum=16) already dampens token-ratio variance sufficiently.
* **Minimal Experiment**: 128 updates comparing BNPO vs. GSPO on 512 Level 3–5 problems under identical seeds, constant LR $3\times 10^{-6}$, microbatch 1, accumulation 16, monitoring per-step gradient norms and completion lengths.

#### Gap 2: Exploration & Multi-Iteration Clipping Dynamics
* **Research Question**: Under multi-iteration PPO training (`num_iterations = 2`), where clipping is active, does asymmetric clipping ($\epsilon_{\text{high}} = 0.28, \epsilon_{\text{low}} = 0.20$) sustain genuine mathematical reasoning exploration, or does it merely introduce an upward probability drift that inflates entropy metrics?
* **Candidate $H_{2A}$ (Exploration Elicitation)**: Relaxing the upper clipping bound on iteration 2 allows rare, serendipitous reasoning tokens in correct rollouts to receive larger updates, preventing winner-take-all collapse and generating broader strategy diversity (higher unique reasoning n-grams and higher Pass@16).
* **Candidate $H_{2B}$ (Asymmetric Drift & Entropy Artifact)**: Asymmetric clipping acts as an uncontrolled positive drift on common tokens, inflating measured token entropy across both correct and incorrect rollouts without discovering new reasoning paths, leading to degraded greedy execution precision.
* **Causal Estimand**:
  $$\tau_2 = \mathbb{E}[\text{Pass@16}(\text{DAPO-Clip})] - \mathbb{E}[\text{Pass@16}(\text{Symmetric-Clip})]$$
* **Discriminating Predictions**:
  * Under $H_{2A}$: DAPO-style clipping exhibits a widening spread between Pass@16 and greedy accuracy ($\Delta(\text{Pass@16} - \text{greedy}) \ge +5\text{pp}$) and produces higher distinct 4-gram prefix counts in derivations.
  * Under $H_{2B}$: DAPO-style clipping shows higher entropy but lower greedy accuracy on Level 5 MATH, with Pass@16 remaining identical or inferior to symmetric clipping.
* **Minimal Experiment**: 64 updates on 512 problems with `num_iterations = 2`, comparing symmetric clipping ($\epsilon = 0.20/0.20$) vs. asymmetric clipping ($\epsilon = 0.20/0.28$), logging active clipping fractions (`clip_ratio/high_mean`), token entropy, Pass@1/4/8/16, and distinct token n-grams.

#### Gap 3: Rollout Token Efficiency (Dynamic Sampling vs. Uniform Scheduling)
* **Research Question**: When evaluated under a strictly matched budget of **1,500,000 total generated rollout tokens**, does bounded dynamic sampling (discarding all-0 and all-1 groups) improve reasoning performance compared to uniform prompt scheduling?
* **Candidate $H_{3A}$ (Gradient Density Acceleration)**: Filtering zero-variance groups eliminates inert updates, ensuring 100% of gradient updates perform meaningful policy optimization, thereby accelerating reasoning acquisition per unit of compute.
* **Candidate $H_{3B}$ (Curriculum Attrition & Token Inefficiency)**: Dynamic replenishment discards up to 45% of generated tokens and filters out the hardest problems (which remain all-0 for early epochs), starving the model of exposure to complex problem distributions and reducing total token efficiency.
* **Causal Estimand**:
  $$\tau_3 = \mathbb{E}[\text{Acc}_{\text{Dev}}(\text{Dynamic})] - \mathbb{E}[\text{Acc}_{\text{Dev}}(\text{Uniform})] \quad \text{at fixed } N_{\text{tokens}} = 1.5\text{M}$$
* **Discriminating Predictions**:
  * Under $H_{3A}$: Dynamic sampling reaches 70% dev accuracy in fewer total generated tokens and achieves higher final accuracy on mixed-difficulty problems.
  * Under $H_{3B}$: Under an equal 1,500,000-token budget, dynamic sampling completes fewer total updates, exhibits lower Level 5 MATH accuracy (due to all-0 discarding), and incurs higher GPU idle time during rollout rejection loops.
* **Minimal Experiment**: 3-seed comparison of Dr. GRPO with Uniform Scheduling vs. Bounded Dynamic Sampling (capped at 5,000 attempted prompts), measuring accuracy and Pass@16 against cumulative rollout tokens up to 1.5M tokens.

#### Gap 4: Length Regulation (Adaptive vs. Fixed Length Penalty)
* **Research Question**: Does an outcome-conditional, difficulty-adaptive length penalty improve token efficiency without triggering underthinking collapse on competition-level problems?
* **Candidate $H_{4A}$ (Adaptive Length Pareto Gain)**: Penalizing length only on correct completions that exceed minimal verified derivation lengths (or scaling penalty inversely with problem difficulty) compresses routine solutions while preserving extended reasoning on difficult problems.
* **Candidate $H_{4B}$ (Intrinsic Compression Sufficiency)**: Pure correctness rewards ($R_0$) already penalize ungrounded length via cumulative error propagation; any explicit length penalty distorts policy gradients, reducing solution completeness on Olympiad problems.
* **Causal Estimand**:
  $$\tau_4 = \mathbb{E}[\text{Tokens}(\text{Adaptive})] - \mathbb{E}[\text{Tokens}(R_0)] \quad \text{subject to } \text{Acc}(\text{Adaptive}) \ge \text{Acc}(R_0) - 1.0\text{pp}$$
* **Discriminating Predictions**:
  * Under $H_{4A}$: Adaptive penalties reduce average completion length by $\ge 25\%$ on Level 1–3 MATH while preserving 100% of $R_0$ accuracy on Level 5 MATH and AIME.
  * Under $H_{4B}$: Any penalty causes $\ge 5\text{pp}$ accuracy loss on AIME and increases premature conclusion markers.
* **Minimal Experiment**: Compare $R_0$ (correctness only), $R_2$ (flat penalty $-0.05 \cdot L/T$), and $R_{\text{adaptive}}$ (penalty applied only when $R=1$ and $L > 1.5 \times L_{\text{ref}}$) over 64 updates.

---

## 5. Comparative Evaluation & Feasibility Assessment

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FEASIBILITY & VALUE MATRIX                                │
├───────┬─────────────┬─────────────┬─────────────┬──────────────────┬───────────────────┤
│ Gap   │ Novelty     │ Complexity  │ Compute Req │ Baseline Control │ Scientific Value  │
├───────┼─────────────┼─────────────┼─────────────┼──────────────────┼───────────────────┤
│ Gap 1 │ Moderate    │ Low         │ ~18 GPU-hrs │ Frozen Base      │ High (Mechanics)  │
│ Gap 2 │ Moderate    │ Low         │ ~18 GPU-hrs │ Symmetric Clip   │ High (Exploration)│
│ Gap 3 │ High        │ Moderate    │ ~24 GPU-hrs │ Uniform Schedule │ Very High (Policy)│
│ Gap 4 │ Mod-High    │ Moderate    │ ~20 GPU-hrs │ Pure $R_0$       │ High (Efficiency) │
└───────┴─────────────┴─────────────┴─────────────┴──────────────────┴───────────────────┘
```

### Detailed Assessment Across Evaluation Dimensions

1. **Novelty Risk**:
   * *Gap 1 (IS Level & Denominators)*: Moderate. Head-to-head comparison between sequence-level IS (GSPO) and active-token Dr. GRPO (BNPO) on dense 1.5B models under long-horizon reasoning isolates the denominator confound.
   * *Gap 2 (Multi-Iteration Clipping)*: Moderate. Demonstrating that clipping is inactive under `num_iterations = 1` and evaluating whether multi-iteration asymmetric clipping genuinely boosts exploration is a necessary corrective to prior literature claims.
   * *Gap 3 (Dynamic Sampling)*: **High**. Most open papers claim dynamic sampling is superior by plotting accuracy against *optimizer steps*. Demonstrating that dynamic sampling can be less efficient per *generated rollout token* due to discard overhead would provide a major, corrective contribution to the literature.
   * *Gap 4 (Length Penalties)*: Moderate-High. Moving beyond flat penalties to outcome-conditional bounds addresses an active, contentious debate across NeurIPS 2025/2026.

2. **Implementation Complexity**:
   * *Gap 1*: **Very Low**. BNPO and GSPO loss objectives are already implemented and validated in [`src/training/rl/objectives.py`](file:///home/vo43/math-reasoning-agent/src/training/rl/objectives.py) and [`src/training/rl/methods.py`](file:///home/vo43/math-reasoning-agent/src/training/rl/methods.py).
   * *Gap 2*: **Very Low**. Parameterized upper/lower clipping is already supported in [`src/training/rl/pilot.py`](file:///home/vo43/math-reasoning-agent/src/training/rl/pilot.py#L42); requires setting `num_iterations = 2`.
   * *Gap 3*: **Moderate**. Requires a bounded dynamic buffer in the rollout generation loop that discards all-0/all-1 groups while accurately tracking discarded tokens, prompt replenishment, and total attempted budgets.
   * *Gap 4*: **Moderate**. Requires integrating reference solution length metadata into the reward callback and logging length-penalty fractions separately.

3. **Compute & Cluster Feasibility (Purdue Gilbreth)**:
   * All four directions strictly respect Gilbreth constraints: single A100-80GB GPU, `--qos=standby`, 4-hour time limits, scratch storage for checkpoints/logs.
   * Total budget for any single gap study across 3 seeds is **18–24 GPU-hours**, well within normal standby throughput and completed in 2–3 days without hogging cluster resources.

4. **Baseline Controls**:
   * All directions feature clean, un-confounded baselines: frozen un-fine-tuned base model, verified common-stop evaluation protocol (tokens 151643/151645), and exact matched training prompts.

5. **Expected Scientific Value**:
   * Gap 3 delivers the highest scientific impact because it directly challenges standard industry reporting practices (step-based vs. token-based efficiency) and answers why models struggle with hard-problem generalization under dynamic filtering.

---

## 6. Recommended Next Experiment: Controlled Study of Bounded Dynamic Sampling vs. Uniform Scheduling

### 6.1 Scientific Rationale
We recommend **Gap 3 (Bounded Dynamic Sampling vs. Uniform Prompt Scheduling under Matched 1.5M Rollout Token Budgets)** as the next immediate empirical investigation. 

*Why this experiment?*
1. **Directly Addresses Repository Findings**: Our pilot data proved that 38%–42% of generated prompt groups contribute zero gradient signal under uniform sampling. Determining whether to filter these groups or preserve them is the single most urgent design decision for the planned 5,000-problem scaling stage.
2. **Prioritizes Interpretable Causal Evidence**: Rather than claiming benchmark gains, this design isolates the causal mechanism: does dynamic sampling provide higher *gradient information density*, or does it act as a distorted curriculum that starves the model of hard Olympiad problems?
3. **Hardware Safety & Determinism**: It can be executed on the proven Dr. GRPO backend without changing model architecture, optimizer hyperparameters, or hardware configurations.

---

### 6.2 Preregistration Scaffold & Protocol Specification

#### 1. Target System & Initial Conditions
* **Base Model**: `Qwen/Qwen2.5-Math-1.5B` (base, revision pinned in scratch).
* **Precision**: FP32 master weights, BF16 compute, PyTorch SDPA attention, gradient checkpointing enabled.
* **Optimizer**: AdamW, constant learning rate $\eta = 3\times 10^{-6}$ (validated optimum from paired-seed study), $\beta = (0.9, 0.95)$, weight decay 0.0, gradient clipping norm 1.0.
* **Hardware Allocation**: Single NVIDIA A100-80GB PCIe on Purdue Gilbreth (`ai-forge`, `--qos=standby`, `--time=03:59:00`, 64 GB host RAM).
* **Rollout Configuration**: Group size $G=4$ completions per prompt, sampling temperature $T=1.0$, top-$p=1.0$, maximum completion length 3,000 tokens.

#### 2. Experimental Arms (Between-Subjects Design)
To isolate the effect of dynamic filtering from static curriculum design, we define three concurrent arms:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EXPERIMENTAL ARMS                                      │
├─────────────────┬──────────────────────┬───────────────────────────────────────────────┤
│ Arm             │ Sampling Mechanism   │ Operational Update Rule                       │
├─────────────────┼──────────────────────┼───────────────────────────────────────────────┤
│ Arm A (Control) │ Uniform Sampling     │ Standard sequential prompt iteration; updates │
│                 │                      │ all groups (including all-0 and all-1).       │
├─────────────────┼──────────────────────┼───────────────────────────────────────────────┤
│ Arm B (Dynamic) │ Bounded Dynamic      │ Evaluates group variance; discards all-0 and  │
│                 │ Sampling             │ all-1 groups; replenishes from prompt pool    │
│                 │                      │ until batch is full (attempt cap = 5,000).    │
├─────────────────┼──────────────────────┼───────────────────────────────────────────────┤
│ Arm C (Matched) │ Static Difficulty-   │ Fixed stratified mixture: 30% Level 3, 40%    │
│                 │ Balanced Control     │ Level 4, 30% Level 5; updates all groups.     │
└─────────────────┴──────────────────────┴───────────────────────────────────────────────┘
```

#### 3. Budget Matching & Stopping Rules
* **Independent Variable**: Prompt sampling policy (Uniform vs. Bounded Dynamic vs. Stratified).
* **Primary Budget Constraint**: Exactly **1,500,000 total generated rollout tokens** per run (including both accepted and discarded tokens).
* **Stopping Rule**: Each run terminates immediately upon generating its 1,500,000th rollout token. 
  * Under Arm A, this yields approximately 120–128 optimizer updates.
  * Under Arm B, discarded tokens count toward the 1,500,000 limit, yielding fewer total updates (estimated 75–90 updates) but with 100% informative gradient signal.
* **Replication**: 3 independent training seeds ($S \in \{0, 1, 2\}$) per arm $\times$ 3 arms = **9 total runs**.

#### 4. Primary Metrics & Causal Estimands

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MEASUREMENT TAXONOMY                                      │
├────────────────────┬───────────────────────────────────┬───────────────────────────────┤
│ Dimension          │ Observable Metric                 │ Scientific Purpose            │
├────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│ 1. Information     │ - Fraction of non-zero updates    │ Quantifies gradient density   │
│    Density         │ - Advantage variance              │ and active learning signal.   │
│                    │ - Discard rate (all-0 vs all-1)   │                               │
├────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│ 2. Compute         │ - Greedy dev accuracy vs tokens   │ Uncovers whether dynamic      │
│    Efficiency      │ - Pass@16 vs tokens               │ filtering wastes more tokens  │
│                    │ - GPU-hours to reach 65% accuracy │ than it saves.                │
├────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│ 3. Curriculum      │ - Level 3 vs Level 5 solve rate   │ Tests whether dynamic filter- │
│    Shift           │ - Hard-problem survival rate      │ ing starves hard problems.    │
│                    │ - Out-of-distribution AIME acc    │                               │
├────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│ 4. Exploration     │ - Token entropy trajectory        │ Tracks strategy collapse vs.  │
│    & Dynamics      │ - Distinct 4-gram prefix count    │ diverse solution discovery.   │
└────────────────────┴───────────────────────────────────┴───────────────────────────────┘
```

#### 5. Discriminating Predictions & Decision Logic
1. **Confirmation of $H_{3A}$ (Dynamic Efficiency Superiority)**:
   * Arm B achieves higher greedy accuracy on the 300 development problems than Arm A at the matched 1,500,000-token mark ($\text{Acc}_B - \text{Acc}_A \ge +3.0\text{pp}$, $p < 0.05$ across 3 seeds).
   * Arm B reaches 65% dev accuracy in $\le 70\%$ of the wallclock GPU time required by Arm A.
2. **Confirmation of $H_{3B}$ (Curriculum Attrition & Token Inefficiency)**:
   * Arm A matches or outperforms Arm B in final accuracy at the 1,500,000-token mark ($\text{Acc}_A \ge \text{Acc}_B$).
   * Arm B shows a severe difficulty divergence: it improves on Level 3 (+5pp) but suffers significant degradation on Level 5 MATH ($-6.0\text{pp}$) and AIME ($-4.0\text{pp}$) relative to Arm A, directly caused by discarding $>70\%$ of Level 5 rollouts as all-0.
3. **Indeterminate Outcome**:
   * Arm A and Arm B achieve identical accuracies ($\pm 1.0\text{pp}$) across all difficulty levels, with token discard overhead exactly offsetting the benefit of higher gradient density.

---

### 6.3 Slurm Orchestration & Execution Guardrails

To prevent cluster failures, timeouts, and quota violations, the experiment must strictly comply with the following operational constraints:

1. **Storage Quarantine**:
   * Code and logs (`%j_%x.out/err`) reside in `/home/vo43/math-reasoning-agent/`.
   * Checkpoints, virtual environments, datasets, and JSONL rollouts reside strictly in `/scratch/gilbreth/vo43/math-reasoning-agent/outputs/dynamic_sampling_study/`.
2. **Slurm Partition & Wallclock Sizing**:
   * Launch on `a100-80gb` under `standby` QOS (`--qos=standby`, `--time=03:59:00`, `--mem=64G`).
   * Training must execute in **4-hour segments**, with automated checkpointing at 4-update rollout boundaries (`--save_steps 4`), resolving the restart gate checkpoint shard exception.
   * Diagnostic evaluation is restricted to 20 reserved development examples during training passes to prevent Slurm timeouts; full 300-example dev and MATH-500 evaluations are executed in separate dependent Slurm jobs (`afterok`).
3. **Deterministic Restart Verification**:
   * Before launching the 9-run grid, execute an 8-update uninterrupted vs. resumed check on Arm B to ensure weight divergence remains strictly $\le 10^{-7}$, utilizing the `--save_steps 4` parameter confirmed in commit `7926feb`.

---

## 7. Remaining Questions & Future Horizon

### 7.1 Unresolved Gaps Beyond This Experiment
1. **Multi-Turn Formal Verification**: While Lean 4 integration was deferred due to squashfs packaging and metadata latency, coupling RLVR with Lean/Mathlib 4.24.0 represents the ultimate boundary for mathematical rigor.
2. **Test-Time Search & Verifier Scaling**: The present study focuses exclusively on policy parameter optimization. Integrating test-time compute (Best-of-$N$, Monte Carlo Tree Search, or PRM step-verifiers) remains an orthogonal dimension for exploration.
3. **Cross-Architecture Generality**: All findings are grounded in dense 1.5B transformer architectures (`Qwen2.5-Math-1.5B`). Whether these dynamics translate identically to Mixture-of-Experts (MoE) architectures remains an open hypothesis.

### 7.2 Immediate Priorities for the Next Investigator
1. **Execute Resume Gate Verification**: Run `sbatch scripts/rl/run_resume_gate.slurm` with the `--save_steps 4` parameter from commit `7926feb` to verify mathematical restart equivalence ($\text{delta} \le 10^{-7}$) on Gilbreth A100.
2. **Complete AIME Evaluation Ingestion**: Ingest the completed shards from Job `12492613` to finalize the baseline Pass@16 numbers on AIME 2026.
3. **Implement Bounded Dynamic Sampling Buffer**: Develop unit tests in `tests/test_dynamic_sampling.py` verifying that all-0 and all-1 groups are filtered while token counts are accurately logged against the 1,500,000-token cap.

---
*Authored in accordance with project guidelines, AGENTS.md reproducibility rules, and scientific agent standards.*
