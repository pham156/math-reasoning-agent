"""Experimental setup preceding the atlas results, grounded in saved run metadata."""


def render_setup(setup, section, prose, table):
    architecture = setup["architecture"]
    section("Experimental Setup: Model and Architecture")
    prose("This document reports the completed Qwen2.5-Math-1.5B study, not the earlier 7B agent prototype. All primary RL comparisons start from the same pretrained base checkpoint, not Qwen-Math-Instruct and not a previously supervised adapter. The unchanged base checkpoint is the no-RL control. The core question is how outcome-reward training, optimization choices and inference-time sampling change mathematical reasoning.")
    table("Model architecture and parameter-update design, verified against the saved model configuration and RL run.",
          ["Component", "Specification"],
          [["Backbone", setup["model_name"] + "; approximately 1.5 billion parameters; decoder-only causal Transformer"],
           ["Transformer depth / hidden width", f"{architecture['num_hidden_layers']} layers / {architecture['hidden_size']} hidden units"],
           ["Attention", f"{architecture['num_attention_heads']} query heads; {architecture['num_key_value_heads']} key/value heads (grouped-query attention)"],
           ["Feed-forward width", str(architecture["intermediate_size"])],
           ["Vocabulary / context window", f"{architecture['vocab_size']:,} tokens / {architecture['max_position_embeddings']:,} positions"],
           ["RL trainable parameters", "Full model; no LoRA adapter and no quantization in the primary RL study"],
           ["Numerical precision", "FP32 trainable/master weights; BF16 forward/backward compute; gradient checkpointing"],
           ["Auxiliary models", "No trained critic, learned reward model or prompt-based verifier in the RL objective; frozen reference policy only in nonzero-KL arms"]], [.29, .71])
    prose(r"\begin{center}\fbox{\begin{minipage}{.91\linewidth}\centering MATH problem $\rightarrow$ Qwen policy $\rightarrow$ $G$ sampled solutions\\Extract final answers $\rightarrow$ rule-based correctness rewards\\Within-problem relative advantages $\rightarrow$ clipped policy update\end{minipage}}\end{center}")
    prose("The policy generates complete reasoning responses, and the reward evaluates only the extracted final answer. Training updates the model parameters; it is not few-shot prompting. The main evaluation pipeline does not insert retrieved solutions, run a verifier/revision loop or perform proof-tree search. Inference ablations reuse trained checkpoints and vary decoding or answer selection without further parameter updates.")
    prose(r"The tokenizer's chat template renders a system message and the problem as a user message. The fixed system instruction is: \texttt{Please reason step by step, and put your final answer within \textbackslash boxed\{\}.} The same core prompt is used for RL and its ordinary evaluation; wording variants are isolated inference ablations.")
    versions = setup["versions"]
    prose("The saved primary runtime records TRL " + versions["trl"] + ", Transformers " + versions["transformers"]
          + ", PyTorch " + versions["torch"] + " and Math-Verify " + versions["math-verify"]
          + ". These identify the implemented backend, not a claim to reproduce every component of an external paper's training recipe.")

    section("Datasets, Development Split and Separation")
    counts = setup["math_original_counts"]
    prose(f"The original MATH benchmark contains {counts['train']:,} training and {counts['test']:,} test problems in our saved snapshot. We filter the original training split to Levels 3--5, require parseable reference answers and prompts no longer than {setup['prompt_cap']:,} tokens, and deduplicate whitespace-normalized problem text. This leaves {setup['eligible']:,} eligible problems. A category/difficulty-balanced development set is reserved first with split seed {setup['split_seed']}; the RL training subset is selected only from the remaining problems.")
    table("Data actually used: development is a custom MATH-train holdout, not MATH-500.",
          ["Dataset / partition", "Source", "Size", "Use"],
          [["RL training", "Original MATH train, eligible Levels 3--5", "512 problems", "Parameter updates; same fixed problem pool across primary RL arms"],
           ["Dev300", "Reserved original MATH train, Levels 3--5", "300 problems", "Hyperparameter comparisons and checkpoint monitoring; no training updates"],
           ["Dev100", "Fixed subset of Dev300", "100 problems", "Sampled exploration, inference-budget, prompt and selection diagnostics"],
           ["MATH-500", "HuggingFaceH4/MATH-500; selected from original MATH test", "500 problems", "External competition-math evaluation"],
           ["GSM8K-100", "Fixed 100-problem subset of GSM8K test", "100 problems", "External arithmetic/word-problem evaluation"],
           ["AIME 2025", "math-ai/aime25 test; AIME I + II", "30 problems", "External hard competition-math evaluation"],
           ["AIME 2026", "math-ai/aime26 test; AIME I + II", "30 problems", "External hard competition-math evaluation"],
           ["Retention probes", "WikiText subset; BoolQ; ARC-Easy", "32 text blocks; 100 questions per task", "Perplexity and raw/chat label-likelihood checks, not full-benchmark claims"]], [.18, .32, .15, .35])
    prose(r"\textbf{Where dev comes from.} Dev300 and Training512 are two disjoint subsets of the original MATH \emph{training} split. MATH-500 comes from the original MATH \emph{test} split. Dev100 is contained within Dev300, so those two diagnostic cohorts are intentionally not independent. The 512-problem pool is the completed pilot/stabilization study; the larger training-data target in the research plan is not presented as completed training here.")
    overlap = setup["overlaps"]
    table("Direct overlap audit of the saved cohorts, using whitespace-normalized problem-text hashes.",
          ["Comparison", "Shared problems"],
          [["Training512 versus Dev300", overlap["train_dev"]], ["Training512 versus MATH-500", overlap["train_math500"]],
           ["Dev300 versus MATH-500", overlap["dev_math500"]]], [.75, .25])
    prose("This audit establishes no exact normalized-text overlap among those saved cohorts; it does not rule out semantic near-duplicates or exposure in the base model's pretraining. Development results guide tuning, and external benchmark results have also been inspected across experiments: external does not mean untouched confirmatory data. Each AIME year contains 15 AIME I and 15 AIME II problems; pooled AIME uses 60 distinct problems, so small accuracy changes correspond to few questions.")

    section("Algorithms and Training Parameters")
    table("Implemented objectives and controls used in the completed comparison.",
          ["Method", "Training signal and update"],
          [["No RL", "Untouched Qwen2.5-Math-1.5B base; identical ordinary evaluation prompt and decoding"],
           ["GRPO", "Within-problem mean-centered, group-standardized rewards; token-level importance ratios and response-length-normalized clipped policy loss"],
           ["Dr. GRPO", "Mean-centered rewards without group-standard-deviation scaling; token-level ratios and fixed maximum-completion-length loss normalization"],
           ["DAPO-style", "Group-standardized rewards; token-level ratios; batch-token loss normalization; asymmetric clipping distances 0.2 / 0.28"],
           ["GSPO", "Group-standardized rewards; sequence-level importance ratios; clipping distances 0.0003 / 0.0004"],
           ["SFT", "Assistant-only cross entropy on human reference solutions; a supervised control, not reinforcement learning"],
           ["RFT", "Assistant-only cross entropy on correctness-verified base-model samples; rejection-sampling fine-tuning, not a new RL objective"]], [.18, .82])
    prose(r"For a group of $G$ sampled solutions, GRPO uses relative advantages proportional to $(r_i-\bar r)/(s_r+\epsilon)$; Dr.~GRPO uses $r_i-\bar r$ without reward-standard-deviation scaling. DAPO-style and GSPO are the implemented objective variants above, not full-paper reproductions. In particular, the DAPO-style arm does not add dynamic sampling. Loss and gradient-norm scales differ across objectives and should not be ranked directly. PPO and REINFORCE are not completed arms of this study.")
    table("Shared RL defaults and the tuned continuation setting.", ["Parameter", "Value / interpretation"],
          [["Initialization and data", "Same base checkpoint and frozen 512-problem MATH pool; no-RL control has no updates"],
           ["Learning rate / schedule", "Original method comparison: 1e-6; tuned GRPO continuation: 3e-6; constant schedule, zero warmup"],
           ["Optimizer", "AdamW; betas 0.9 / 0.95; weight decay 0; gradient-norm clipping at 1.0"],
           ["Prompt / response caps", f"{setup['prompt_cap']:,} / {setup['completion_cap']:,} tokens"],
           ["Training generation", "G=4 independent responses per problem; temperature 1.0; top-p 1.0; top-k disabled"],
           ["Rollout batch", f"16 distinct prompts x 4 responses = {setup['generation_batch_responses']} generated responses"],
           ["Optimizer batch", f"Microbatch 1; accumulation {setup['optimizer_batch_responses']}; {setup['optimizer_batch_responses']} responses per update"],
           ["Update budget", "128 updates per pass over the 512 problems at G=4; continuation checkpoints at 256 and 512 total updates"],
           ["Clipping / KL", "GRPO/Dr. GRPO: 0.2 / 0.2; DAPO-style: 0.2 / 0.28; GSPO: 0.0003 / 0.0004; KL beta 0 unless ablated; one iteration per generation batch"],
           ["Reference run", f"Training seed {setup['training_seed']}; split seed {setup['split_seed']}; reference-run presentation, not cross-seed averages"]], [.27, .73])
    prose("The default 64-response rollout batch supplies four 16-response optimizer updates; it is not a batch of 64 distinct problems. Extending to 512 updates revisits the same training pool for four passes rather than introducing 2,048 distinct training problems. Group-size ablations retain the same 512 problems but change generated responses and updates, so they are not compute-matched.")

    section("Ablation Design and Evaluation Protocol")
    table("Actual ablation axes; parameters not named in an arm remain at its stated reference setting.",
          ["Study", "Values / controlled change"],
          [["Learning rate", "All four algorithms: 2e-7, 5e-7, 1e-6; GRPO and Dr. GRPO additionally 2e-6, 3e-6; 128-update comparison"],
           ["KL / clipping", "GRPO at LR 1e-6: KL beta 0, 0.001 or 0.01; symmetric clip distance 0.1, 0.2 or 0.3; one factor changed at a time"],
           ["Reward design", "GRPO at LR 3e-6: correctness only; correctness + format bonus; correctness - length penalty"],
           ["Training exploration", "At LR 3e-6: temperature 0.7, 1.0 or 1.3 with G=4; G=2, 4 or 8 with temperature 1.0"],
           ["Training duration", "Tuned GRPO at 128, 256 and 512 total updates; same 512-problem pool"],
           ["Inference only", "Greedy caps 512 / 1,024 / 2,048 / 3,000; eight-candidate temperatures 0.4 / 0.7 / 1.0; saved-candidate budgets 1 / 2 / 4 / 8; isolated prompt-wording and selection-rule changes"]], [.23, .77])
    prose(r"Let $C$ be binary final-answer correctness, $F$ indicate exactly one nonempty parseable boxed answer, and $L$ be completion length. The rewards are $R_0=C$, $R_{\rm format}=C+0.10F$, and $R_{\rm length}=C-0.05\min(L/3000,1)$. Correctness uses the existing final-answer extractor, whitespace-normalized equality and Math-Verify equivalence fallback; extraction/verification failures receive zero. It is not a learned verifier score or a proof-validity reward. Raw correctness and shaping components are logged separately.")
    sft = setup["sft"]
    names = {"gt_full_original": "Original full SFT", "gt_full": "Full SFT", "rft_full": "Full RFT", "gt_lora": "LoRA SFT",
             "rft_full_clean": "Full RFT, filtered", "rft_lora_clean": "LoRA RFT, filtered"}
    table("Supervised follow-up configuration: target responses are not necessarily distinct source problems.",
          ["Control", "Training targets", "LR", "Epochs", "Opt. batch", "Adaptation"],
          [[name, sft[key]["train_n"], f"{sft[key]['learning_rate']:g}", sft[key]["epochs"],
            sft[key]["gradient_accumulation_steps"],
            "LoRA r=64, alpha=128" if "lora" in sft[key] else "Full parameters"] for key, name in names.items()],
          [.24, .16, .12, .09, .11, .28])
    prose(r"The supervised follow-ups use a cosine schedule, 3\% warmup, microbatch 1 with accumulation 16, a 4,096-token total sequence cap and assistant-only labels ending at the chat-end token. The original historical full-SFT control instead uses one epoch, LR 1e-5 and optimizer batch 4; it is not pooled with the three-epoch follow-up. LoRA targets attention and MLP projection layers and is merged for evaluation. RFT target counts can exceed 512 because a source problem can contribute multiple verified responses; filtering reduces 967 to 655 targets. Only independently complete evaluation cohorts are scored.")
    prose("Ordinary benchmark decoding is greedy (temperature 0), cap 3,000, with common end-of-text/chat-end stopping. Sampled diagnostics use eight responses per problem and the explicitly labeled temperature. External consensus fixes temperature 0.7 and uses a strict majority of normalized nonempty final answers, falling back to the first sampled candidate when no majority exists; no verifier chooses candidates. Retrospective subset/order averages are reported separately from fixed-order external selection.")
    prose("Reported outcomes distinguish greedy accuracy, sampled pass@1, oracle pass@k and selected-answer accuracy. Additional measurements cover loss, reward components/variance, entropy, KL, clipping, truncation, answer diversity, response length and generated-token cost. Paired bootstrap intervals resample problems, not training seeds. Historical stored grading and the later strict closed-box audit are never silently merged.")
