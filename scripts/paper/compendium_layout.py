"""Human-facing comparisons; detailed provenance belongs in the JSON archive."""

from collections import defaultdict

from scripts.paper.prepare_compendium import escape, longtable, pct


MODELS = ("base_no_rl", "step128", "step256", "step512")
METHODS = ("grpo", "dr_grpo", "dapo_style", "gspo")
NAMES = {"base_no_rl": "No RL", "base": "No RL", "baseline": "No RL",
         "step128": "GRPO 128", "step256": "GRPO 256", "step512": "GRPO 512",
         "grpo": "GRPO", "dr_grpo": "Dr. GRPO", "dapo_style": "DAPO-style", "gspo": "GSPO", "sft": "Original SFT"}


def label(name):
    return NAMES.get(name, name)


def decimal(value, digits=4):
    return "--" if value is None else f"{value:.{digits}f}"


def render(records, setup):
    groups = defaultdict(list)
    for record in records:
        groups[record["section"]].append(record)

    def rows(section):
        return groups[section]

    def index(section):
        return {(r["label"], r["cohort"]): r["metrics"] for r in rows(section)}

    text = []

    def section(title):
        if text:
            text.append(r"\clearpage")
        text.append(r"\section{" + escape(title) + "}")

    def prose(value):
        text.append(value)

    def table(title, headers, values, widths=None):
        content = longtable(headers, values, widths)
        text.append(content.replace("\n" + r"\toprule ", "\n" + r"\caption{" + escape(title)
                                    + r"}\\" + "\n" + r"\toprule ", 1))

    def figure(filename, caption, width=r"\linewidth"):
        text.append(r"\begin{center}\includegraphics[width=" + width + "]{figures/" + filename
                    + r"}\par\captionof{figure}{" + escape(caption) + r"}\end{center}")

    from scripts.paper.compendium_setup import render_setup
    render_setup(setup, section, prose, table)

    transfer = index("Checkpoint transfer")
    initial = index("Initial external transfer")
    controls = {r["label"]: r["metrics"] for r in rows("Initial controls")}
    duration = {r["metrics"]["step"]: r["metrics"] for r in rows("Training duration")}
    aime = index("AIME 2025 and 2026")
    section("Benchmark Results")
    prose("The following results use the model, frozen splits and protocols defined above. Accuracy is in percent; development, external benchmarks, sampled oracle availability and selected-answer performance remain separately labeled.")
    values = []
    for model in MODELS:
        dev = controls["baseline"] if model == "base_no_rl" else duration[int(model[4:])]
        values.append([label(model), pct(dev["accuracy"]), pct(transfer[model, "math500"]["accuracy"]),
                       pct(transfer[model, "gsm8k100"]["accuracy"]), pct(aime[model, "pooled greedy"]["accuracy"])])
    for model in METHODS:
        values.append([label(model) + " original", pct(controls[model]["accuracy"]),
                       *[pct(initial[model, b]["accuracy"]) for b in ("math500", "gsm8k100")], "--"])
    table("Consolidated benchmarks: tuned GRPO uses LR 3e-6; original methods use LR 1e-6 and 128 updates.",
          ["Model / updates", "Dev300", "MATH-500", "GSM8K-100", "AIME-60"], values, [.28, .18, .18, .18, .18])
    prose(r"\textbf{Main finding.} GRPO improves MATH-500 from 34.8\% to 70.4\% and GSM8K-100 from 38\% to 85\%. Continuing from 128 to 512 updates improves these external scores despite development accuracy falling from 75\% to 69\%. Development and external trajectories should therefore be read separately, not collapsed into one learning curve.")
    prose("Greedy decoding uses a 3,000-token completion cap and common stopping conventions. Sampled pass@1 averages candidate correctness; pass@8 measures whether any candidate is correct, not the accuracy of an implementable selector. Completed negative results are retained. Dashes indicate an unavailable complete cohort, never zero accuracy. Reused controls are not independent replications. The tables preserve historical grading; a separate grading audit appears later.")

    section("Training Dynamics")
    figure("initial_training.pdf", "Matched initial GRPO and Dr. GRPO runs at learning rate 1e-6: training correctness, surrogate loss, token entropy and response length. Curves summarize observed training updates, not held-out performance.", r".92\linewidth")
    figure("extended_training.pdf", "Tuned GRPO at learning rate 3e-6 through 512 updates. Pale loss traces are raw observations; darker curves are trailing means. Policy-surrogate loss is not a supervised prediction loss and need not decrease monotonically.", r".92\linewidth")

    section("Numerical Training Dynamics")
    histories = [(label(method) + " original", controls[method].get("training_diagnostics", {})) for method in METHODS]
    histories += [(f"Tuned GRPO {step}", m.get("training_diagnostics", {})) for step, m in duration.items()]
    values = []
    for name, stats in histories:
        loss = stats.get("loss", {})
        values.append([name, *["--" if loss.get(key) is None else f"{loss[key]:.3e}"
                               for key in ("mean", "std", "first_quarter", "last_quarter")]])
    table("Policy-surrogate loss: numerical summaries of observed optimizer-update logs.",
          ["Training run", "Mean loss", "Loss SD", "First-quarter mean", "Last-quarter mean"], values, [.28, .16, .16, .20, .20])
    values = []
    for name, stats in histories:
        entropy, length = stats.get("entropy", {}), stats.get("completions/mean_length", {})
        values.append([name, *[decimal(entropy.get(key), 3) for key in ("mean", "first_quarter", "last_quarter")],
                       *[decimal(length.get(key), 0) for key in ("mean", "first_quarter", "last_quarter")]])
    table("Exploration and response-length dynamics, measured on training generations.",
          ["Training run", "Mean entropy", "Early entropy", "Late entropy", "Mean tokens", "Early tokens", "Late tokens"], values,
          [.25, .12, .12, .12, .13, .13, .13])
    table("Cumulative exposure and training signal during the tuned continuation.",
          ["Updates", "Train correctness", "Generated M tokens", "Mean gradient norm"],
          [[step, pct(m["training_correctness"]), f"{m['training_completion_tokens']/1e6:.3f}",
            decimal(m.get("training_diagnostics", {}).get("grad_norm", {}).get("mean"), 3)] for step, m in duration.items()])
    prose("Means use the available observations of each metric through the stated update, excluding step-zero initialization. Early/late values average the first/last quarter of those observations. Loss SD is variation across logged updates, not uncertainty across independent training runs. GRPO and Dr. GRPO use different loss normalization, so their absolute loss and gradient-norm scales are not directly comparable. Continuation rows are nested cumulative prefixes, not independent runs.")

    section("Held-Out Trajectories and Exploration")
    figure("heldout_accuracy.pdf", "Measured complete checkpoint cohorts only. Gaps mark unavailable scheduled evaluations; no partial benchmark is scored or interpolated.")
    values = []
    exploration = {r["label"]: r["metrics"] for r in rows("Initial exploration")}
    for model in ("baseline", *METHODS):
        m = exploration[model]
        values.append([label(model) + " (original)", *[pct(m["pass_at_k"][str(k)]) for k in (1, 4, 8)], f"{m['avg_tokens']:.0f}"])
    for step, m in duration.items():
        sampled = next(r["metrics"] for r in rows("Inference temperature and token budget")
                       if r["metrics"]["checkpoint"] == f"step{step}" and r["metrics"]["mode"] == "sampled"
                       and r["metrics"]["temperature"] == 1)
        values.append([f"Tuned GRPO {step}", *[pct(m["pass_at_k"][str(k)]) for k in (1, 4, 8)], f"{sampled['avg_candidate_tokens']:.0f}"])
    table("Fixed dev100: eight sampled responses at temperature 1; output length is the mean per sampled response.",
          ["Policy", "Sampled@1", "Pass@4", "Pass@8", "Response tokens"], values, [.34, .16, .16, .16, .18])
    prose("RL changes both the probability of obtaining a correct answer and the distribution of candidate solutions. Oracle pass@8 is consistently higher than sampled@1, leaving room for selection, but availability alone does not establish that voting or a verifier will identify the correct candidate.")

    section("Optimizer Ablations")
    rates = {(r["metrics"]["algorithm"], r["metrics"]["learning_rate"]): r["metrics"] for r in rows("Learning-rate controls")}
    table("Learning-rate sweep: dev300 greedy accuracy after 128 updates, fixed reference runs.",
          ["Algorithm", "2e-7", "5e-7", "1e-6", "2e-6", "3e-6"],
          [[label(a), *[pct(rates.get((a, lr), {}).get("accuracy")) for lr in (2e-7, 5e-7, 1e-6, 2e-6, 3e-6)]] for a in METHODS])
    values = []
    for algorithm in METHODS:
        for lr in (2e-7, 5e-7, 1e-6, 2e-6, 3e-6):
            if (algorithm, lr) not in rates:
                continue
            history = rates[algorithm, lr]["history_summary"]
            values.append([label(algorithm), f"{lr:g}", decimal(history["reward_mean"], 3),
                           decimal(history["reward_std"], 3), decimal(history["entropy_mean"], 3),
                           decimal(history["grad_norm_mean"], 3), decimal(history["response_length_mean"], 0)])
    table("Learning-rate ablation: measured training behavior, not just final answer accuracy.",
          ["Algorithm", "LR", "Mean reward", "Reward SD", "Entropy", "Gradient norm", "Response tokens"], values,
          [.20, .12, .13, .13, .12, .15, .15])
    section("KL and Clipping Ablations")
    values = []
    for r in rows("KL and clipping controls"):
        m = r["metrics"]
        s = m["settings"]
        values.append([f"{s['learning_rate']:g}", f"{s['kl_beta']:g}", f"{s['clip_lower']:g}/{s['clip_upper']:g}",
                       decimal(m["history"]["kl_mean"], 5), decimal(m["fixed_prefix_reference_kl"], 4),
                       decimal(m["clip_fraction_mean"], 6), decimal(m["history"]["entropy_mean"], 3), pct(m["accuracy"])])
    table("KL/clipping ablation: online KL, offline drift and observed clipping fraction.",
          ["LR", "KL beta", "Clip bounds", "Train KL", "Prefix KL", "Clip fraction", "Entropy", "Dev300"], values,
          [.13, .10, .13, .13, .13, .13, .12, .13])
    prose("GRPO's highest observed development accuracy is at LR 3e-6; Dr. GRPO's is at LR 2e-6. This does not establish a universal ordering: Dr. GRPO exceeds GRPO at LR 2e-6. Prefix KL is an offline shared-context diagnostic, not the online training KL penalty.")
    dr = index("Matched-LR Dr. GRPO transfer")
    table("External matched-LR comparison at LR 3e-6, 128 updates.", ["Method", "MATH-500", "GSM8K-100", "AIME-60"],
          [["GRPO", pct(transfer["step128", "math500"]["accuracy"]), pct(transfer["step128", "gsm8k100"]["accuracy"]), pct(aime["step128", "pooled greedy"]["accuracy"])],
           ["Dr. GRPO", *[pct(dr["dr_grpo_lr3e-06", b]["accuracy"]) for b in ("math500", "gsm8k100", "aime")]]])
    prose("The Dr. GRPO external row evaluates the matched LR 3e-6 setting, not its development-best LR 2e-6 checkpoint. No unmeasured external score is inferred.")
    prose("Training reward SD is the source summary's variation across logged reward means; it is not within-group reward variance. Missing online KL means the trainer did not record it, not zero policy drift. Clip fraction is reported as a fraction. Gradient norms depend on each objective's normalization; they should be interpreted within an algorithm.")
    prose("For GRPO, raising LR from 2e-7 to 3e-6 increases mean training reward from 0.276 to 0.501 while mean entropy falls from 0.871 to 0.379 and mean response length falls from 1,096 to 926 tokens. These measurements show the exploration/optimization trade-off behind the accuracy sweep, rather than only its winning setting.")

    section("Reward Design and Training Exploration")
    values = []
    for r in rows("Reward design"):
        m = r["metrics"]
        values.append([{"correctness": "Correctness only", "format_bonus": "Correctness + format", "length_penalty": "Correctness - length"}.get(r["label"], r["label"]),
                       pct(m["accuracy"]), f"{m['avg_generated_tokens']:.0f}", pct(m["training"]["correctness_mean"]),
                       f"{m['training']['completion_tokens']/1e6:.2f}"])
    table("Reward study at LR 3e-6: correctness is logged independently of the shaped reward.",
          ["Reward", "Dev300", "Greedy tokens", "Train correctness", "Train M tokens"], values, [.32, .15, .17, .19, .17])
    values = []
    for r in rows("Reward design"):
        m = r["metrics"]["training"]
        name = {"correctness": "Correctness only", "format_bonus": "+ Format bonus", "length_penalty": "+ Length penalty"}[r["label"]]
        values.append([name, decimal(m["reward_mean"], 4), decimal(m["correctness_group_variance"], 4),
                       decimal(m["total_group_variance"], 4), pct(m["format_mean"]),
                       pct(m["format_only_failure_fraction"]), decimal(m["length_penalty_mean"], 5)])
    table("Reward-design mechanism: raw correctness variance versus shaped-reward variance.",
          ["Reward design", "Shaped mean", "Correctness variance", "Shaped variance", "Format %", "Format-only wrong %", "Mean penalty"], values,
          [.24, .12, .14, .13, .11, .15, .11])
    prose(r"The format arm adds $0.10F$ to binary correctness $C$; the length arm uses $C-0.05\min(L/3000,1)$. Shaped reward means are not directly comparable across reward definitions. Neither auxiliary reward improves the reference run's development accuracy.")
    values = []
    for r in rows("Training exploration"):
        m = r["metrics"]
        s = m["settings"]
        values.append([f"{s['temperature']:g}", s["num_generations"], pct(m["accuracy"]),
                       pct(m["pass_at_k"]["1"]), pct(m["pass_at_k"]["8"]), f"{m['training']['completion_tokens']/1e6:.2f}"])
    table("Training temperature/group size, evaluated with common inference settings.",
          ["Train T", "Group size", "Dev300", "Sampled@1", "Pass@8", "Train M tokens"], values)
    control_tokens = next(r["metrics"]["training"]["completion_tokens"] for r in rows("Training exploration")
                          if r["label"] == "correctness")
    table("Training-sampling ablation: exposure, reward signal and generated length.",
          ["Train T", "Group size", "Updates", "Responses", "Mean tokens", "Train correctness", "Reward variance", "Cost x"],
          [[f"{r['metrics']['settings']['temperature']:g}", r["metrics"]["settings"]["num_generations"],
            r["metrics"]["training"]["responses"] // 16, r["metrics"]["training"]["responses"],
            decimal(r["metrics"]["training"]["response_length"], 0), pct(r["metrics"]["training"]["correctness_mean"]),
            decimal(r["metrics"]["training"]["correctness_group_variance"], 4),
            decimal(r["metrics"]["training"]["completion_tokens"] / control_tokens, 2)]
           for r in rows("Training exploration")], [.09, .10, .10, .12, .14, .16, .16, .13])
    prose("Group-size conditions match 512 unique training problems, not optimizer updates or generated tokens: group sizes 2, 4 and 8 use 64, 128 and 256 updates respectively. Higher sampled accuracy can therefore come with substantially different training cost; this is not a compute-matched comparison.")
    prose("Reward variance is the mean within-prompt variance across generated candidate rewards, not the variance of the reward curve. Cost is generated training tokens relative to the T=1, group-size-4 control; it is not a wall-clock speed ratio. Format-only wrong is the percentage of all training responses that are incorrect yet receive a format bonus.")
    prose(r"The format bonus raises mean shaped reward to 0.5989, but 30.32\% of training responses are incorrect and still receive that bonus; higher shaped reward does not imply higher held-out accuracy. Training at T=1.3 generates about 2.17 times the reference tokens while only 6.64\% of sampled training responses are correct. These are sampling-distribution measurements, not greedy evaluation scores.")

    section("Supervised and Rejection-Sampling Fine-Tuning")
    sft = index("Supervised and rejection-sampling fine-tuning")
    names = {"gt_full_original": "Original full SFT", "gt_full": "Full SFT", "gt_lora": "LoRA SFT",
             "rft_full": "Full RFT", "rft_full_clean": "Full RFT, filtered", "rft_lora_clean": "LoRA RFT, filtered"}
    values = []
    for model, name in names.items():
        scores = [sft.get((model, b), {}).get("accuracy") for b in ("dev300", "math500", "gsm8k100", "aime")]
        if model == "gt_full_original":
            scores[0] = controls["sft"]["accuracy"]
        m = sft.get((model, "dev300"), {})
        values.append([name, *map(pct, scores), f"{m['avg_generated_tokens']:.0f}" if "avg_generated_tokens" in m else "--"])
    table("All independently complete SFT/RFT cohorts; unfinished benchmark cohorts are omitted, not prorated.",
          ["Adaptation", "Dev300", "MATH-500", "GSM8K-100", "AIME-60", "Dev tokens"], values, [.30, .14, .14, .14, .14, .14])
    prose("SFT imitates human solutions; rejection-sampling fine-tuning (RFT) imitates retained correct model-generated solutions. These use the same source problem pool, but the retained RFT target set is not identical to the human-target training set. Full-parameter follow-ups use LR 5e-6 with a cosine schedule for three epochs; LoRA follow-ups use rank 64, alpha 128 and LR 1e-4. The original SFT control is a separate historical configuration.")
    prose(r"\textbf{Interpretation.} RFT improves substantially over human-solution SFT on the complete development and MATH-500 cohorts, but the full RFT MATH-500 score (61.8\%) remains below tuned GRPO. Filtering and parameterization are not uniformly better across available cohorts. Longer RFT responses are not evidence of more efficient reasoning.")
    table("Mean completion length on complete external cohorts.",
          ["Adaptation", "MATH-500 tokens", "GSM8K-100 tokens"],
          [[names[m], *[f"{sft[m,b]['avg_generated_tokens']:.0f}" if (m,b) in sft else "--" for b in ("math500", "gsm8k100")]] for m in names], [.44, .28, .28])
    prose("Original full SFT already uses the corrected common stopping convention in the benchmark tables above. The following audit separates the historical stopping bug from remaining answer errors and nontermination.")

    stop = {r["label"]: r["metrics"] for r in rows("SFT stopping correction")}
    if stop:
        table("Same original SFT checkpoint on the same 300 development problems, before and after stopping correction.",
              ["Stopping convention", "Correct /300", "Accuracy %", "Truncated /300", "Truncation %", "Mean tokens"],
              [[name, m["correct"], pct(m["accuracy"]), m["truncated"], pct(m["truncation_rate"]), decimal(m["avg_generated_tokens"], 1)]
               for name, m in stop.items()], [.30, .14, .14, .14, .14, .14])
        m = stop["Common EOS + chat end"]
        prose(f"The correction fixes {m['fixed_vs_original']} answers and breaks {m['broken_vs_original']} against the original saved evaluation; no model was retrained. It reduces wasted continuation substantially but does not erase the SFT accuracy deficit. These are existing saved evaluations, not new generations or a new grading convention.")
    names = {"base": "No RL", "grpo": "GRPO LR 1e-6", "lr3": "GRPO LR 3e-6, 128", "step512": "GRPO LR 3e-6, 512",
             "sft": "Original SFT, corrected", "sft_new": "Follow-up full SFT", "rft": "Full RFT"}
    table("Common-stop dev300 outcomes: truncation and incorrect answers overlap; counts retain historical saved grading.",
          ["Policy", "Wrong /300", "Truncated /300", "Truncated correct", "Truncated wrong", "Nontruncated acc. %"],
          [[names[r["label"]], m["wrong"], m["truncated"], m["truncated_correct"], m["truncated_wrong"], pct(m["nontruncated_accuracy"])]
           for r in rows("Common-stop completion audit") for m in [r["metrics"]]], [.28, .13, .14, .14, .14, .17])
    prose("A response can contain a correct answer before hitting the token cap. Conversely, ending normally does not imply correct reasoning. Nontruncated accuracy conditions on a different model-dependent subset and must not replace all-problem benchmark accuracy or be treated as a matched comparison. Remaining SFT truncations already occur with the corrected stops; another EOS configuration change is not an established fix.")

    section("AIME Transfer by Year")
    values = []
    for model in MODELS:
        values.append([label(model), *[pct(aime[model, c]["accuracy"]) for c in ("AIME 2025 greedy", "AIME 2026 greedy", "pooled greedy")],
                       pct(aime[model, "pooled sampled mean@8/pass@8"]["mean_at_8"]), pct(aime[model, "pooled sampled mean@8/pass@8"]["pass_at_8"])])
    table("AIME: greedy accuracy and sampled mean@8 / oracle pass@8, pooled over both years.",
          ["Policy", "2025 /30", "2026 /30", "Pooled /60", "Mean@8", "Pass@8"], values, [.25, .15, .15, .15, .15, .15])
    values = []
    for model in MODELS:
        m = aime[model, "pooled greedy"]
        values.append([label(model), pct(m["truncated"] / m["n"]), decimal(m["avg_generated_tokens"], 0)])
    table("AIME generation behavior on the complete 60-problem greedy cohort.",
          ["Policy", "Truncation %", "Mean response tokens"], values)
    prose("AIME remains difficult: pooled greedy gains represent only a few problems, and oracle pass@8 is not a selected-answer score. Small year-specific differences should not be interpreted as reliable algorithm rankings.")

    section("Inference Budget and Temperature")
    inference = [r["metrics"] for r in rows("Inference temperature and token budget")]
    idx = {(m["checkpoint"], m["mode"], m["temperature"], m["cap"]): m for m in inference}
    checkpoints = ("base", "step128", "step256", "step512")
    table("Fixed dev100 greedy accuracy: completion-budget control.", ["Policy", "512 tokens", "1,024", "2,048", "3,000"],
          [[label(m), *[pct(idx[m, "greedy", 0, cap]["first_accuracy"]) for cap in (512, 1024, 2048, 3000)]] for m in checkpoints])
    values = []
    for model in checkpoints:
        for t in (.4, .7, 1.):
            m = idx[model, "sampled", t, 3000]
            values.append([label(model), t, pct(m["selected_accuracy"]),
                           f"{100*(m['selected_accuracy']-m['first_accuracy']):+.1f}", m["missed_correct"],
                           pct(m["truncation_rate"]), f"{m['avg_candidate_tokens']:.0f}", f"{m['avg_problem_tokens']:.0f}"])
    table("Inference temperature: voting gains versus the first sampled candidate, missed oracle opportunities and cost.",
          ["Policy", "T", "Vote %", "Vote gain pp", "Missed correct /100", "Truncation %", "Response tokens", "Group tokens"], values,
          [.17, .06, .11, .13, .14, .13, .13, .13])
    prose("A 2,048-token cap matches the 3,000-token cap's aggregate greedy accuracy for all four policies on this dev100 set. This does not prove per-problem equivalence or transfer to longer external problems. Candidate voting can recover correct answers, but requires generating all eight candidates.")

    section("Prompt Robustness and External Consensus")
    follow = [r["metrics"] for r in rows("Prompt and external consensus controls")]
    table("System-instruction sensitivity on fixed dev100, greedy cap 3,000.", ["Policy", "Original", "Reworded", "Concise"],
          [[label(m), pct(idx[m, "greedy", 0, 3000]["first_accuracy"]),
            *[pct(next(r["selected_accuracy"] for r in follow if r.get("checkpoint") == m and r.get("prompt_variant") == v and r.get("dataset") == "development")) for v in ("reworded", "concise")]] for m in checkpoints])
    values = []
    for record in rows("Prompt and external consensus controls"):
        m = record["metrics"]
        if m.get("dataset") == "development":
            continue
        model = m.get("checkpoint", "step512")
        benchmark = m.get("dataset", "math500")
        control = transfer["base_no_rl" if model == "base" else model, benchmark]
        values.append(["MATH-500" if benchmark == "math500" else "GSM8K-100", label(model), pct(control["accuracy"]),
                       pct(m["selected_accuracy"]), m["fixed_by_consensus"], m["broken_by_consensus"],
                       f"{m['avg_problem_tokens']/control['avg_generated_tokens']:.2f}"])
    table("Complete external consensus: frozen temperature 0.7, eight candidates, original prompt.",
          ["Benchmark", "Policy", "Greedy", "Vote", "Fixes*", "Breaks*", "Token cost x"], values, [.20, .15, .13, .13, .11, .11, .17])
    prose("*Fixes and breaks compare voting against the first sampled candidate, not the greedy control. Token cost is total completion tokens divided by that policy's greedy completion cost. The recovered step512 MATH-500 condition is included only after all 500 problems and eight candidates per problem passed raw-output validation.")
    prose("Prompt changes affect the no-RL control and RL checkpoints differently. External voting gains are small relative to their completion-token cost: high development oracle accuracy should not be substituted for external selected accuracy.")

    section("Candidate Selection and Answer Diversity")
    saved = rows("Saved-candidate selection budgets")
    values = []
    for model in checkpoints:
        for t in (.4, .7, 1.):
            prefix = f"{model} T={t:g} "
            candidates = { (r["metrics"]["k"], r["metrics"]["rule"]): r["metrics"] for r in saved if r["label"].startswith(prefix)}
            first = candidates[1, "first"]["accuracy"]
            vote = candidates[8, "majority_first"]["accuracy"]
            values.append([label(model), t,
                           *[f"{100*(candidates[k, 'majority_first']['accuracy']-first):+.2f}" for k in (2, 4, 8)],
                           f"{100*(candidates[8, 'plurality']['accuracy']-vote):+.2f}",
                           f"{100*(candidates[8, 'first']['oracle_accuracy']-vote):.2f}"])
    table("Candidate-budget ablation: marginal gains in percentage points, rather than repeated absolute accuracies.",
          ["Policy", "T", "Vote 2 gain", "Vote 4 gain", "Vote 8 gain", "Plurality extra 8", "Oracle gap 8"], values,
          [.20, .06, .14, .14, .14, .16, .16])
    prose("These tables reuse saved candidates and average uniformly over subsets and candidate order. They are not new generations or independent replications. In particular, retrospective plurality is a different protocol from the fixed-order strict-majority selector used for external benchmarks. Oracle availability is an upper bound, not a deployable selection rule.")
    values = []
    for r in rows("Answer-distribution diagnostics"):
        m = r["metrics"]
        values.append([label(r["label"].split()[0]), r["label"].split("=")[1], f"{m['answer_entropy_nats']:.3f}",
                       f"{m['unique_nonempty_answers']:.2f}", pct(m["modal_answer_mass"]), pct(m["empty_answer_rate"])])
    table("Final-answer diversity (not token-policy entropy), eight candidates per problem.",
          ["Policy", "T", "Entropy (nats)", "Unique answers", "Modal mass", "Empty"], values)

    section("Selective Coverage and Subgroup Results")
    values = []
    for model in checkpoints:
        for t in (.4, .7, 1.):
            found = {(r["metrics"]["k"], r["metrics"]["rule"]): r["metrics"] for r in saved if r["label"].startswith(f"{model} T={t:g} ")}
            m = found[8, "majority_abstain"]
            values.append([label(model), t, *[pct(found[k, "majority_abstain"]["coverage"]) for k in (2, 4, 8)],
                           pct(m["answered_accuracy"]), pct(m["accuracy"])])
    table("Strict-majority abstention: coverage and accuracy must be reported together.",
          ["Policy", "T", "Coverage 2", "Coverage 4", "Coverage 8", "Answered acc. 8", "All-problem acc. 8"], values,
          [.19, .06, .14, .14, .14, .16, .17])
    values = []
    for dimension in ("by_type", "by_level"):
        base = transfer["base_no_rl", "math500"][dimension]
        for group in sorted(base):
            before, after = base[group], transfer["step512", "math500"][dimension][group]
            values.append([group, f"{100*(after['accuracy']-before['accuracy']):+.1f}",
                           pct(before["truncated"] / before["n"]), pct(after["truncated"] / after["n"]),
                           decimal(before["avg_generated_tokens"], 0), decimal(after["avg_generated_tokens"], 0)])
    table("MATH-500 subgroup behavior: no-RL to GRPO 512, measured on identical complete problem cohorts.",
          ["Subgroup", "Gain pp", "Base trunc. %", "RL trunc. %", "Base tokens", "RL tokens"], values,
          [.30, .10, .15, .15, .15, .15])

    section("Grading Sensitivity and General Retention")
    audit = {r["label"]: r["metrics"] for r in rows("Historical strict-grading audit")}
    models = {"base": "No RL", "grpo_1e-6": "GRPO original", "drgrpo_1e-6": "Dr. GRPO original",
              "grpo_3e-6_128": "GRPO 128", "grpo_3e-6_256": "GRPO 256", "grpo_3e-6_512": "GRPO 512",
              "sft_gt_full_original": "Original full SFT", "sft_gt_full": "Full SFT"}
    table("Grading audit: each cell is lenient / strict closed-box accuracy (November 20 snapshot).", ["Policy", "Dev300", "MATH-500", "GSM8K-100", "AIME greedy"],
          [[name, *[f"{pct(audit[b+'/'+m]['lenient'])} / {pct(audit[b+'/'+m]['strict'])}" if b+'/'+m in audit else "--" for b in ("dev300", "math500", "gsm8k100", "aime_greedy")]] for m, name in models.items()],
          [.26, .185, .185, .185, .185])
    values = []
    for r in rows("Non-math retention"):
        if r["label"] not in ("baseline", *METHODS):
            continue
        m = r["metrics"]
        tasks = {(x["task"], x["format"]): x for x in m["tasks"]}
        values.append([label(r["label"]), f"{m['wikitext_subset_perplexity']:.2f}",
                       *[pct(tasks[t, f]["accuracy"]) for t in ("boolq", "arc_easy") for f in ("raw", "chat")]])
    table("Original LR 1e-6 policies: auxiliary retention controls.",
          ["Policy", "WikiText PPL", "BoolQ raw", "BoolQ chat", "ARC raw", "ARC chat"], values)
    prose("WikiText perplexity uses 32 fixed 512-token blocks; classification probes use 100 questions each. Raw versus chat formatting changes label likelihoods substantially, so these narrow diagnostics do not establish broad capability preservation. The strict-box audit changes the extraction convention, not historical training rewards, and final-answer correctness is not proof verification.")

    section("Paired Improvements and Evidence Scope")
    values = []
    for r in rows("Paired uncertainty estimates"):
        m = r["metrics"]
        if m["model"] != "grpo_3e-6_512" or m["baseline"] != "base":
            continue
        result = m["lenient"]
        name = models.get(m["model"], m["model"])
        base = models.get(m["baseline"], m["baseline"])
        cohort = {"dev300": "Dev300", "math500": "MATH-500", "gsm8k100": "GSM8K-100",
                  "aime_greedy": "AIME greedy", "aime_sampled": "AIME sampled"}[r["cohort"]]
        values.append([cohort, f"{name} vs {base}", f"{100*result['difference']:+.1f}",
                       f"[{100*result['ci95'][0]:+.1f}, {100*result['ci95'][1]:+.1f}]", result["improved"], result["worsened"]])
    table("Consolidated final-checkpoint gains: paired bootstrap differences, historical lenient grading.",
          ["Cohort", "Comparison", "Delta", "95% CI", "Fixed", "Broken"], values, [.16, .37, .09, .20, .09, .09])
    prose("Intervals resample paired problems, not training seeds; they do not measure training-run uncertainty and are not corrected for multiple comparisons. Main tables use a fixed reference run rather than selecting favorable seeds. Complete results with worse accuracy remain visible; failed or partial cohorts never receive a score.")
    prose("The accompanying JSON preserves exact metrics and source hashes. Older unavailable 7B/retrieval outputs are not reconstructed from conversation history. No new training, generation or cluster jobs are needed to build this document.")
    from scripts.paper.compendium_supplement import render_supplement
    render_supplement(records, setup, section, prose, table)
    return "\n\n".join(text) + "\n"
