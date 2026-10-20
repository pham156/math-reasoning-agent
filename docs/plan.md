# Reinforcement Learning Post-Training for Mathematical Reasoning

## 1. Project Overview

Recent large language models have shown substantial improvements in mathematical reasoning through reinforcement learning (RL) post-training. Mathematics is particularly suitable for RL because many problems have objectively verifiable answers, allowing models to receive correctness-based rewards without requiring human preference labels.

However, successful mathematical RL depends on more than simply applying an RL algorithm. Recent literature suggests that several factors may affect reasoning performance:

* the RL optimization algorithm;
* the model's ability to explore successful solutions;
* the difficulty and selection of training problems;
* reward design;
* rollout and sampling strategy;
* the pretrained model used to initialize RL;
* the relationship between supervised fine-tuning and RL;
* the amount of training and inference-time computation.

The goal of this project is to conduct a controlled empirical study of these factors and determine **what makes RL post-training effective for mathematical reasoning**.

Rather than committing to a single algorithm or dataset at the beginning, we will first establish a common experimental framework and then investigate several representative design choices.

---

# 2. Main Research Question

> **What algorithmic, data, reward, and exploration choices determine the effectiveness of reinforcement learning post-training for mathematical reasoning?**

The overall pipeline is:

```text
Pretrained Mathematical Model
            |
            v
     Mathematical Problems
            |
            v
      Generate Rollouts
            |
            v
     Verifiable Rewards
            |
            v
       RL Optimization
            |
            v
       Updated Policy
            |
            v
 Mathematical Evaluation
```

We are interested in both **final mathematical performance** and **what happens during this process**.

---

# 3. Research Questions

## RQ1 — RL Algorithm

> How does the choice of RL optimization method affect mathematical reasoning?

We will investigate representative algorithms from different RL post-training families.

Possible candidates include:

### PPO Family

PPO constrains policy updates and commonly uses a learned value function or critic for advantage estimation.

It is historically important in language-model RL and provides a useful comparison with newer reasoning-specific approaches.

### REINFORCE Family

Policy-gradient approaches directly increase the probability of trajectories receiving higher rewards.

Possible variants include:

* REINFORCE;
* REINFORCE++;
* RLOO-style methods.

These methods are conceptually simpler and can provide useful policy-gradient baselines.

### GRPO Family

GRPO generates groups of solutions for each problem and estimates relative advantages using rewards within the group.

It avoids training a separate critic and has become widely used in mathematical RLVR.

### Modern GRPO / RLVR Variants

Possible candidates include:

* Dr. GRPO;
* DAPO-style methods;
* other recent modifications to advantage estimation, clipping, normalization, or sampling.

The final algorithm set will be selected after evaluating implementation complexity, compute requirements, and compatibility with the existing Open-R1 infrastructure.

We expect to select approximately **2–3 representative methods rather than implementing every available algorithm**.

---

# 4. Hypothesis 1 — RL Algorithms Produce Different Learning Dynamics

Even when starting from the same model and using the same mathematical problems and rewards, different optimization methods may behave differently.

We will compare:

* final mathematical accuracy;
* reward progression;
* training stability;
* policy divergence;
* response length;
* exploration/diversity;
* sample efficiency;
* computational cost.

The objective is not simply to determine which algorithm obtains the highest score.

Instead, we want to understand:

> **How does the optimization method change the way mathematical reasoning is learned?**

---

# 5. RQ2 — Exploration and Initial Model Capability

RL can only reinforce successful reasoning trajectories if the model can discover them.

Before performing RL, we will sample multiple solutions from the frozen initial policy.

For example:

```text
Problem A:
✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

Problem B:
✓ ✗ ✗ ✓ ✗ ✓ ✗ ✓

Problem C:
✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗
```

These correspond roughly to:

```text
Already mastered
      |
      v
Mixed / learnable
      |
      v
Currently too difficult
```

For each training problem, estimate:

```text
p_correct =
number of correct sampled solutions
-----------------------------------
number of sampled solutions
```

This provides a **model-relative measure of difficulty**.

---

# 6. Hypothesis 2 — Useful Exploration Determines RL Effectiveness

We hypothesize that RL may be most effective when the model can occasionally discover a correct solution but does not solve the problem reliably.

Extremely easy problems may provide little information because almost every trajectory receives the same positive reward.

Extremely difficult problems may also provide little information because almost every trajectory receives zero reward.

Therefore:

> **Problems near the model's current capability boundary may provide the strongest RL training signal.**

This hypothesis will be tested rather than assumed.

---

# 7. RQ3 — Training Data Selection

Training problems will be analyzed based on both their original dataset difficulty and the model's empirical success rate.

Possible training subsets include:

```text
Easy for the model

Intermediate / mixed-success

Hard for the model

Random
```

Controlled RL runs can then compare these subsets while keeping other training settings fixed.

This allows us to ask:

> What types of mathematical problems are actually useful for RL training?

rather than assuming that harder problems are automatically better.

---

# 8. Dynamic Data Selection and Curriculum Learning

If model-relative difficulty is found to matter, we will investigate whether it can be used to improve RL training.

Instead of uniformly sampling problems:

```text
Dataset
   |
random sample
   |
   v
  RL
```

we could use:

```text
Current Policy
      |
      v
Estimate success rate
      |
      v
Select informative problems
      |
      v
RL update
      |
      v
Policy improves
      |
      v
Re-estimate difficulty
```

Problems that become too easy could gradually receive less training weight, while harder problems could enter the curriculum as the model improves.

This would create an **adaptive, model-relative mathematical curriculum**.

It will be treated as a potential improvement based on the results of the initial experiments rather than assumed to work beforehand.

---

# 9. RQ4 — Reward Design

The simplest RLVR reward is final-answer correctness:

```text
correct answer   → positive reward
incorrect answer → zero reward
```

This will serve as the primary reward because it is objective and easy to verify.

However, the literature suggests several possible reward designs.

## Outcome Reward

Reward only the final mathematical answer.

Advantages:

* simple;
* objective;
* low risk of reward-model errors.

Disadvantage:

* sparse credit assignment.

## Process Reward

Provide signals about intermediate reasoning steps.

Potential advantage:

* better credit assignment.

Potential challenge:

* intermediate mathematical reasoning is more difficult to verify reliably.

## Additional Reward Components

Possible secondary components include:

* answer-format reward;
* length regularization;
* partial correctness;
* structured reasoning reward.

We will first establish a correctness-only baseline before testing more complicated reward designs.

---

# 10. RQ5 — Sampling and Rollout Design

Most basic RLVR implementations generate multiple independent solutions:

```text
Problem
 ├── Solution 1
 ├── Solution 2
 ├── Solution 3
 └── Solution 4
```

However, alternative rollout strategies may provide more efficient exploration.

Possible designs include:

### Independent Sampling

Generate complete independent solutions.

### Shared-Prefix / Tree Sampling

Allow multiple reasoning trajectories to share an initial reasoning prefix before branching:

```text
             Problem
                |
          Reasoning Prefix
           /           \
       Branch A       Branch B
       /    \          /    \
     A1     A2       B1     B2
```

This could allow more targeted exploration of different reasoning decisions.

### Adaptive Rollout Allocation

Use fewer samples on problems that are clearly easy and more samples on uncertain or difficult problems.

Sampling design is particularly interesting because rollout generation is one of the major computational costs of mathematical RL.

This will be considered a secondary or stretch experiment depending on available time.

---

# 11. RQ6 — RL vs Supervised Fine-Tuning

Our existing project already contains supervised/QLoRA experiments, which gives us an opportunity to compare different post-training paradigms.

Possible comparison:

```text
              Initial Model
                   |
          +--------+--------+
          |                 |
         SFT                RL
          |                 |
          +--------+--------+
                   |
               Evaluation
```

Potentially:

```text
Initial Model
     |
    SFT
     |
     RL
     |
Evaluation
```

Questions include:

> Does RL improve reasoning differently from supervised imitation of solution traces?

> Does RL generalize differently from SFT?

> Does SFT followed by RL behave differently from direct RL?

This comparison is useful because SFT teaches the model to imitate provided reasoning, while RL allows the model to generate its own reasoning and reinforces successful trajectories.

SFT will therefore serve primarily as a **post-training baseline**, not the main focus of the project.

---

# 12. RQ7 — Starting Policy

RL results may depend strongly on the capabilities already present in the initial model.

Possible starting points include:

```text
Base mathematical model

vs.

Instruction-tuned mathematical model
```

A base model may provide different exploration behavior than an instruction-tuned model.

An instruction-tuned model may already contain stronger structured reasoning behavior but could also have a more concentrated output distribution.

If compute permits, we can compare how the same RL procedure behaves from different starting policies.

This is a secondary experiment rather than a requirement.

---

# 13. RQ8 — Sharpening vs Discovery

One important question in recent reasoning-RL research is whether RL creates genuinely new reasoning capabilities or primarily increases the probability of capabilities that already existed in the pretrained model.

We can investigate this using pre-RL sampling.

For each evaluation problem, generate many solutions from the initial model.

Suppose:

```text
Initial greedy answer: WRONG

but

Initial pass@16:
one sampled solution is CORRECT
```

After RL:

```text
RL greedy answer: CORRECT
```

This suggests RL may have **sharpened an existing capability**.

The correct reasoning trajectory already existed in the initial policy's distribution, but RL made it more likely.

Compare this with:

```text
Initial pass@16: 0 correct

RL model: correct
```

which provides stronger evidence that RL enabled reasoning behavior that was not easily discoverable before training.

---

# 14. Sharpening/Discovery Analysis

For each evaluation problem, record:

```text
baseline greedy correctness

baseline pass@k

RL greedy correctness

RL pass@k
```

Then classify improvements.

Example:

```text
Baseline greedy wrong
Baseline sampled correct
RL greedy correct

→ likely sharpening
```

versus:

```text
Baseline greedy wrong
Baseline sampled never correct
RL correct

→ possible capability expansion/discovery
```

We cannot prove from finite sampling that a capability did not exist in the original model, so conclusions must be stated carefully.

Nevertheless, this analysis can provide useful empirical evidence about **what RL is actually changing**.

---

# 15. RQ9 — RL and Test-Time Scaling

Another important question is how RL interacts with additional inference-time computation.

Evaluate both baseline and RL-trained models using:

```text
1 sample
4 samples
8 samples
possibly more samples
```

Then compare their pass@k curves.

Possible outcome:

```text
RL improves greedy accuracy
but
baseline and RL have similar pass@8
```

This would suggest that RL mainly concentrates probability on good solutions.

Another possibility:

```text
RL improves both greedy accuracy
and
pass@8
```

which could suggest a broader improvement in reasoning capability.

This experiment connects RL post-training with test-time scaling.

---

# 16. Model

The preferred model family is:

**Qwen2.5-Math**

A smaller model can be used for:

* debugging;
* smoke tests;
* rapid RL experiments.

A larger model can be used for selected final experiments if compute permits.

For controlled comparisons, the same initial model should be used across experimental conditions.

Our existing mathematical reasoning project already provides substantial infrastructure for this model family.

---

# 17. Training Data

The exact dataset will be selected after examining compatibility with the Open-R1 pipeline and answer verification.

Candidate sources include established mathematical reasoning datasets and verified mathematical problem collections.

For RLVR, the essential training information is:

```text
Problem
+
Reference Answer
```

The model generates its own reasoning trajectory.

We do not need a human-written reasoning trace for the basic RL experiment.

Important dataset properties include:

* verifiable answers;
* mathematical diversity;
* range of difficulty;
* sufficient number of problems;
* low contamination with evaluation benchmarks.

---

# 18. Evaluation Data

Training and evaluation sets must remain separate.

Evaluation should contain:

## Standard Mathematical Reasoning

A sufficiently large benchmark for reliable overall comparison.

## Difficulty-Specific Evaluation

Problems covering different levels of mathematical difficulty.

## Hard Generalization

More difficult mathematical problems that are outside the primary RL training distribution.

The exact benchmarks can be finalized after dataset selection and contamination checks.

---

# 19. Core Evaluation Metrics

## Mathematical Performance

* final-answer accuracy;
* performance by difficulty;
* performance by mathematical category.

## Sampled Performance

* pass@k;
* solution diversity.

## RL Training

* mean reward;
* reward variance;
* policy divergence;
* entropy;
* training stability.

## Reasoning Behavior

* response length;
* correct-response length;
* incorrect-response length;
* self-correction patterns where measurable.

## Efficiency

* training time;
* generated tokens;
* number of rollouts;
* compute requirements.

---

# 20. Fixed/Broken Analysis

Compare every trained policy against the initial model.

Each problem belongs to one of four categories:

```text
Baseline correct → RL correct

Baseline correct → RL wrong
                   BROKEN

Baseline wrong → RL correct
                 FIXED

Baseline wrong → RL wrong
```

Report:

```text
number fixed
number broken
net change
```

Then analyze fixed/broken problems by:

* difficulty;
* category;
* initial pass@k;
* model-relative difficulty;
* response length.

This can help determine **what kinds of reasoning RL improves or damages**.

---

# 21. Training Dynamics

Every RL experiment should log behavior throughout training rather than evaluating only the final checkpoint.

Track:

```text
training reward
held-out accuracy
response length
KL divergence
entropy
reward variance
fraction of successful rollouts
fraction of mixed-reward groups
```

This allows us to identify situations such as:

```text
reward ↑
accuracy ↑
```

versus:

```text
reward ↑
accuracy →
```

or:

```text
reward ↑
accuracy ↓
```

These outcomes provide very different conclusions about RL training.

---

# 22. Scaling

Once stable configurations have been identified, we can investigate scaling along selected dimensions.

Possible dimensions include:

### Training Data

Small → medium → larger training sets.

### Training Duration

Short → medium → longer RL training.

### Rollout Budget

Few → moderate → many trajectories per problem.

### Model Size

Smaller → larger models within the same family, if resources permit.

We should change one major dimension at a time rather than performing a large Cartesian hyperparameter search.

---

# 23. Experimental Structure

The project should proceed sequentially.

## Stage 1 — Infrastructure and Literature

* Audit both repositories.
* Review candidate RL algorithms.
* Select compatible data.
* Standardize answer verification.
* Standardize evaluation.
* Establish baseline model.

---

## Stage 2 — Initial Policy Analysis

Before RL:

* evaluate baseline accuracy;
* generate multiple rollouts;
* measure pass@k;
* estimate problem-level success probabilities;
* analyze exploration;
* characterize model-relative difficulty.

This stage should happen before expensive RL training.

---

## Stage 3 — RL Algorithm Study

Select approximately 2–3 representative RL methods.

Train them under controlled conditions.

Compare:

* learning curves;
* accuracy;
* stability;
* compute;
* reasoning behavior.

---

## Stage 4 — Data and Exploration Study

Construct training subsets based on model-relative difficulty.

Compare training on:

```text
easy
intermediate
hard
random
```

Determine whether initial exploration predicts subsequent RL effectiveness.

---

## Stage 5 — Reward Study

Using a stable RL configuration:

* establish correctness-only reward;
* test selected reward modifications if useful;
* analyze whether reward changes improve actual reasoning.

---

## Stage 6 — Deeper Analysis

Perform:

* RL vs SFT comparison;
* sharpening vs discovery analysis;
* pass@k/test-time scaling analysis;
* fixed/broken analysis.

These experiments help explain **what changed**, not merely how much benchmark accuracy changed.

---

## Stage 7 — Improvement

Based on earlier results, select one promising improvement.

Possible directions:

### Difficulty-Aware Curriculum

Train near the model's current capability boundary.

### Dynamic Problem Sampling

Continuously adjust the training distribution as the model improves.

### Adaptive Rollout Allocation

Spend generation compute on problems where additional exploration is useful.

### Tree / Shared-Prefix Sampling

Explore reasoning branches more efficiently.

### Improved Reward

Use additional reliable signals beyond terminal correctness.

We should choose the improvement based on evidence from earlier experiments rather than deciding it in advance.

---

## Stage 8 — Final Evaluation

Run the selected final models on held-out mathematical benchmarks.

Perform:

* deterministic evaluation;
* sampled/pass@k evaluation;
* difficulty analysis;
* fixed/broken analysis;
* sharpening/discovery analysis;
* efficiency comparison.

---

# 24. Recommended Core Experiments

Because the complete design space is too large to exhaustively explore, the **core project** should focus on:

### Core Experiment A — RL Algorithm

Compare representative RL post-training approaches.

### Core Experiment B — Exploration and Data

Determine whether model-relative problem difficulty predicts RL effectiveness.

### Core Experiment C — Training Dynamics

Determine whether increasing reward corresponds to increasing held-out reasoning performance.

### Core Analysis D — Sharpening vs Discovery

Determine whether RL primarily increases the probability of solutions already discoverable by the initial policy.

These four components form a coherent research story.

---

# 25. Secondary Experiments

Depending on time and compute:

* reward design;
* RL vs SFT;
* base vs instruct starting policy;
* test-time scaling;
* curriculum learning;
* dynamic problem sampling;
* adaptive rollout allocation;
* tree/shared-prefix sampling;
* model/data scaling.

These should be selected based on results from the core experiments.

---

# 26. Existing Infrastructure

Our current mathematical reasoning project already contains:

```text
Qwen mathematical inference
answer extraction
answer normalization
benchmark evaluation
multi-sample generation
consensus
verifier infrastructure
retrieval infrastructure
QLoRA/SFT experiments
initial GRPO/RLVR experiments
fixed/broken analysis
```

Relevant components can be reused:

```text
answer normalization
       ↓
RL correctness reward

multi-sample inference
       ↓
exploration + pass@k

existing benchmarks
       ↓
pre/post RL evaluation

QLoRA experiments
       ↓
SFT comparison

fixed/broken analysis
       ↓
RL behavior analysis
```

Retrieval and consensus do not need to be part of the core RL study.

---

# 27. Open-R1 Integration

The teammate's Open-R1-based repository can provide the primary RL training infrastructure.

Before experiments begin, audit:

```text
supported algorithms
model configuration
reward functions
dataset format
prompt template
rollout generation
LoRA/full training
checkpoint format
logging
distributed training
vLLM integration
```

The two projects should ideally be connected as:

```text
        Existing Math Project
        /        |          \
       /         |           \
evaluation   exploration   analysis
       \         |           /
        \        |          /
          Shared Interface
                |
                |
             Open-R1
                |
        RL Training Pipeline
```

A checkpoint produced by Open-R1 should be directly evaluable using the existing mathematical reasoning pipeline.

---

# 28. Minimum Viable Project

If time or compute becomes constrained:

```text
1 mathematical model
        +
2 representative RL approaches
        +
1 verifiable correctness reward
        +
exploration analysis
        +
difficulty-based data experiment
        +
held-out evaluation
        +
sharpening/discovery analysis
```

This is already a complete RL research project.

It answers:

1. How does RL algorithm choice affect mathematical reasoning?
2. What types of problems provide useful RL signal?
3. Does initial exploration predict RL improvement?
4. Is RL discovering new solutions or sharpening existing capabilities?

---

# 29. Full Project Design

The complete research space can be summarized as:

```text
                         Mathematical RL
                               |
       +-----------------------+-----------------------+
       |                       |                       |
   Optimization             RL Signal              Exploration
       |                       |                       |
 PPO / REINFORCE          Outcome reward        Temperature
 GRPO / variants          Process reward        Rollout count
       |                  Reward shaping        Tree sampling
       |                       |                Adaptive sampling
       +-----------------------+-----------------------+
                               |
                         Training Data
                               |
                +--------------+--------------+
                |              |              |
            Difficulty      Diversity      Curriculum
                |              |              |
                +--------------+--------------+
                               |
                         RL Training
                               |
                  +------------+------------+
                  |                         |
             Performance                 Behavior
                  |                         |
             Accuracy                  Length
             pass@k                    Diversity
             Generalization            Policy change
                  |                         |
                  +------------+------------+
                               |
                    Sharpening vs Discovery
```

---

# 30. Potential Final Research Story

The exact final story should depend on the experimental results.

One possible progression is:

### Observation

Different mathematical problems provide very different amounts of useful RL signal.

### Analysis

Initial-policy exploration predicts which problems benefit from RL.

### Algorithm Study

Different RL algorithms respond differently to these reward distributions.

### Interpretation

RL improvements are partly explained by sharpening reasoning trajectories that already exist in the initial policy.

### Intervention

Use this information to improve training through dynamic problem selection, curriculum learning, adaptive exploration, or another evidence-driven modification.

This gives the project a progression of:

```text
Literature
    ↓
Observation
    ↓
Hypothesis
    ↓
Controlled Experiment
    ↓
Analysis
    ↓
Intervention
    ↓
Evaluation
```

rather than simply:

```text
run RL
↓
report accuracy
```

The broader goal is to understand **when, why, and how RL post-training improves mathematical reasoning**, and then use that understanding to design a more effective training procedure.
