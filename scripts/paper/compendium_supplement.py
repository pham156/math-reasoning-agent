"""Readable supplementary comparisons, without filenames or metric-path dumps."""

from collections import defaultdict
from pathlib import PurePosixPath

from scripts.paper.prepare_compendium import pct


AUDIT_FAMILIES = {
    "aime_checkpoints_20261115": "AIME checkpoints",
    "checkpoint_ablations_20261113": "Inference temperature and completion budget",
    "checkpoint_benchmarks_20261106": "External checkpoint benchmarks",
    "checkpoint_followup_20261115": "Prompt and external consensus follow-ups",
    "controls_exploration_20261014": "Common-stop original controls",
    "dense_replay_20261115": "Intermediate checkpoint evaluations",
    "dev300_sft_20261013": "Historical development and original SFT evaluations",
    "external_benchmarks_20261015": "Original external algorithm comparison",
    "grpo_controls_20261017": "KL and clipping ablations",
    "grpo_extended_20261105": "Extended GRPO",
    "grpo_lr_upper_20261016": "Higher-learning-rate GRPO",
    "lr_tuning_20261014": "Lower-learning-rate algorithm sweep",
    "reward_exploration_20261024": "Reward and training-exploration ablations",
    "sft_study_20261115": "Supervised follow-up evaluations",
}


def readable(value):
    replacements = (("rft_lora_clean", "LoRA RFT filtered"), ("rft_full_clean", "Full RFT filtered"),
                    ("gt_full_original", "Original full SFT"), ("gt_full", "Full SFT"),
                    ("rft_full", "Full RFT"), ("gt_lora", "LoRA SFT"),
                    ("dapo_style", "DAPO-style"), ("dr_grpo", "Dr. GRPO"),
                    ("base_no_rl", "No RL"), ("baseline", "No RL"), ("base_", "No RL "),
                    ("gspo", "GSPO"), ("grpo", "GRPO"), ("gsm8k100", "GSM8K-100"),
                    ("math500", "MATH-500"), ("dev300", "Dev300"), ("development", "Dev100"), ("aime", "AIME"),
                    ("sft", "SFT"), ("clip_0p1", "clip=0.1"), ("clip_0p3", "clip=0.3"), ("kl_", "KL beta="),
                    ("reworded", "reworded prompt"), ("concise", "concise prompt"),
                    ("original", "original prompt"), ("temperature_0p7", "train T=0.7"),
                    ("temperature_1p3", "train T=1.3"), ("rollouts_2", "train G=2"),
                    ("rollouts_8", "train G=8"), ("format_bonus_seed0", "format bonus"),
                    ("length_penalty_seed0", "length penalty"), ("step-", "GRPO "),
                    ("step", "GRPO "), ("_lr", " LR="), ("_t", " T="), ("_cap", " cap="))
    # These are labels encoded by the campaign schema, not numeric data parsing.
    for before, after in replacements:
        value = value.replace(before, after)
    return " ".join(value.replace("_", " ").split())


def audit_label(value):
    parts = PurePosixPath(value).parts
    family = parts[0]
    if family not in AUDIT_FAMILIES:
        raise ValueError(f"Unknown grading-audit family: {family}")
    if family.startswith("controls_exploration"):
        return readable(parts[-1]) + " / " + parts[1]
    if family.startswith("external_benchmarks"):
        return readable(parts[-1]) + " / " + readable(parts[1])
    if family.startswith("dev300_sft"):
        return readable(parts[-1]) + " / greedy"
    if family.startswith("checkpoint_ablations") or family.startswith("checkpoint_followup"):
        return readable(parts[1])
    if family.startswith("aime_checkpoints"):
        return readable(parts[2]) + " / " + parts[4]
    if family.startswith("checkpoint_benchmarks"):
        return readable(parts[2]) + " / " + (readable(parts[3]) if parts[1] == "evaluations" else "Dev100")
    if family.startswith("dense_replay"):
        return readable(parts[2]) + " / " + readable(parts[3])
    if family.startswith("grpo_extended"):
        return readable(parts[2]) + " / " + parts[3]
    if family.startswith("sft_study"):
        return readable(parts[2]) + " / " + readable(parts[3])
    if family.startswith("reward_exploration"):
        if parts[1] == "control_evaluations":
            return "Correctness control / sampled"
        return readable(parts[3]) + " / " + parts[4]
    return readable(parts[2])


def render_supplement(records, setup, section, prose, table):
    from scripts.paper.compendium_layout import METHODS, decimal, label
    groups = defaultdict(list)
    for row in records:
        groups[row["section"]].append(row)

    section("Learning-Signal Diagnostics")
    table("Original dev100 exploration: distribution of correct candidates across eight sampled solutions.",
          ["Policy", "All wrong %", "Mixed correctness %", "All correct %", "Sampled truncation %"],
          [[label(r["label"]), *[pct(r["metrics"][key]) for key in ("all_wrong_rate", "mixed_rate", "all_correct_rate", "truncation_rate")]]
           for r in groups["Initial exploration"]], [.24, .18, .22, .18, .18])
    rows = []
    rates = sorted(groups["Learning-rate controls"], key=lambda r: (METHODS.index(r["metrics"]["algorithm"]), r["metrics"]["learning_rate"]))
    for record in rates:
        m = record["metrics"]
        history = m["history_summary"]
        rows.append([label(m["algorithm"]), f"{m['learning_rate']:g}", decimal(history["reward_group_std_mean"], 4),
                     pct(history["zero_variance_group_rate"]), pct(history["mixed_reward_group_rate"])])
    table("Training learning signal across all completed learning-rate arms.",
          ["Algorithm", "LR", "Mean group reward SD", "Zero-variance groups %", "Mixed-reward groups %"], rows,
          [.21, .13, .22, .22, .22])
    prose("The first table describes eight-sample evaluation groups; the second describes training groups, normally four responses per prompt. Mixed means some candidates are correct and others wrong, not a mixture of reward definitions. Zero-variance groups have zero centered reward advantage, but optimizer momentum or an explicit KL term can still change parameters. Group reward SD is not the standard deviation of the training reward curve. DAPO-style here retains the fixed prompt pool rather than dynamically resampling informative groups.")

    section("Checkpoint and Restart Validation")
    rows = []
    for proof in setup["restart_checks"]:
        name = "Extended GRPO" if proof["method"] == "extended_grpo" else label(proof["method"])
        rows.append([name, f"{proof['boundary']} -> {proof['target']}", "identical" if proof["same_rollouts"] else "different",
                     f"{proof['max_parameter_difference']:.1e}", f"{proof['tolerance']:.1e}", "passed" if proof["passed"] else "failed"])
    table("Saved uninterrupted-versus-resumed proof checks, not new benchmark runs.",
          ["Algorithm", "Restart / final update", "Rollouts", "Max weight difference", "Tolerance", "Validation"], rows,
          [.20, .20, .15, .19, .13, .13])
    table("Reconstructed intermediate checkpoints: narrower rollout/metric-prefix evidence.",
          ["Checkpoint", "Rollout prefix", "Training metric prefix", "Original weight-hash equality"],
          [[p["step"], "identical" if p["rollouts_identical"] else "different", "matched" if p["training_metrics_matched"] else "different", "Unavailable: original intermediate weights rotated"]
           for p in setup["replay_checks"]], [.13, .20, .22, .45])
    prose("The original four-algorithm gates restart after update 4 and compare at update 8; the extended GRPO gate crosses a repeated-data-pass boundary, restarting at 8 and comparing at 16. Before continuation, the original gates also check checkpoint-boundary equality. Saved proofs require matching sampled completions, parameter keys, deterministic configuration and a maximum weight difference no larger than 1e-7. These are implementation checks, not evidence of better reasoning. Historical failed attempts are not reclassified as passed; only the successful replacement proofs are tabulated.")
    prose("The reconstructed 192/256 checkpoints match original rollout and logged-training prefixes. Their original intermediate weight files were rotated, so no direct original-weight hash equality is claimed. This weaker reconstruction evidence remains separate from the full restart-equivalence gates.")

    section("Supplement: Inference Subgroup Ablations")
    prose("All 120 saved category/level breakdowns are shown below. These use the same fixed dev100 problems and eight-candidate, fixed-order strict-majority/first-candidate-fallback inference. They are not MATH-500 subgroup evaluations or retrospective subset/order averages. Voting gain is relative to the first sampled candidate; oracle gap is availability minus selected accuracy, not a deployable gain. Small subgroups have correspondingly coarse percentages.")
    subgroups = groups["Inference subgroup diagnostics"]
    for dimension, title in (("by_type", "Category"), ("by_level", "Difficulty")):
        for temperature in (.4, .7, 1.):
            rows = []
            selected = [r for r in subgroups if r["cohort"].startswith(dimension + ":") and r["label"].endswith(f"T={temperature:g}")]
            for r in sorted(selected, key=lambda r: (r["cohort"], r["label"])):
                m = r["metrics"]
                policy = r["label"].split()[0]
                rows.append([r["cohort"].split(": ", 1)[1], label(policy), m["n"], pct(m["selected_accuracy"]),
                             f"{100*(m['selected_accuracy']-m['first_accuracy']):+.1f}",
                             f"{100*(m['pass_at_8']-m['selected_accuracy']):.1f}", pct(m["truncation_rate"]),
                             decimal(m["avg_candidate_tokens"], 0)])
            table(f"{title} ablation at inference temperature {temperature:g}: complete dev100 subgroup metrics.",
                  [title, "Policy", "Problems", "Vote %", "Gain pp", "Oracle gap pp", "Truncation %", "Response tokens"], rows,
                  [.25, .15, .08, .10, .10, .12, .10, .10])

    section("Supplement: Extraction and Grading Diagnostics")
    prose("The following 14 tables cover all 123 independently complete response-level audit groups in the November 20 snapshot. Denominators count responses: a 100-question, eight-sample run contributes 800, not 100. Strict correct requires the audit's closed-box extraction; lenient-only counts accepted answers that strict extraction rejects. Multi-box means multiple distinct boxed answers, not all malformed outputs. Truncation percentages are per response. Stored mismatches compare historical labels to the audit's re-extracted lenient labels and do not silently replace training rewards.")
    audits = defaultdict(list)
    for r in groups["Response-level grading diagnostics"]:
        audits[PurePosixPath(r["label"]).parts[0]].append(r)
    if set(audits) != set(AUDIT_FAMILIES):
        raise ValueError("Grading supplement campaign coverage changed")
    for family, title in AUDIT_FAMILIES.items():
        rows = []
        for r in audits[family]:
            m = r["metrics"]
            n = m["samples"]
            rows.append([audit_label(r["label"]), n, m["strict_correct"], m["lenient_only"],
                         pct(m["truncated"] / n), m["truncated_strict_correct"], m["multiple_distinct_boxes"],
                         m["stored_mismatch_hash_fix"] + m["stored_mismatch_unexplained"]])
        table(title + ": response-level extraction audit.",
              ["Policy / protocol", "Responses", "Strict correct", "Lenient only", "Trunc. %", "Trunc. strict correct", "Multi-box", "Stored mismatch"], rows,
              [.32, .10, .10, .09, .09, .11, .09, .10])
    prose("Stored mismatch sums the audit's two stored-label mismatch counters. These counters and closed-box criteria describe extraction reliability, not algebraic or proof-step validation. Reused controls and historical pre/post-stopping cohorts are kept in separate tables rather than pooled as independent experiments. Complete audits from unfinished campaigns are included only when their response denominator is complete; partial cohorts and a later recovered result do not retroactively change this dated snapshot.")
