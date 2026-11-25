"""Offline, source-checked qualitative comparisons of saved development responses."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from src.core.config import cache_dir, data_dir, ensure_scratch_path, output_dir, project_root


POLICIES = {
    "base": ("No RL", "controls_exploration_20261014/greedy/baseline"),
    "grpo": ("GRPO, LR 1e-6", "controls_exploration_20261014/greedy/grpo"),
    "dr_grpo": ("Dr. GRPO, LR 1e-6", "controls_exploration_20261014/greedy/dr_grpo"),
    "sft": ("Original human-target SFT, common stops", "controls_exploration_20261014/greedy/sft"),
    "sft_new": ("Follow-up human-target SFT", "sft_study_20261115/evaluations/gt_full/dev300/greedy/sft"),
    "rft": ("Correct-solution RFT", "sft_study_20261115/evaluations/rft_full/dev300/greedy/sft"),
    "lr3": ("GRPO, LR 3e-6, 128 updates", "grpo_lr_upper_20261016/evaluations/grpo_lr3e-06/greedy/grpo"),
    "step512": ("GRPO, LR 3e-6, 512 updates", "grpo_extended_20261105/evaluations/step512/greedy/grpo"),
    "format": ("GRPO, format bonus", "reward_exploration_20261024/reward/evaluations/format_bonus_seed0/greedy/grpo"),
    "length": ("GRPO, length penalty", "reward_exploration_20261024/reward/evaluations/length_penalty_seed0/greedy/grpo"),
    "kl": ("GRPO, KL beta 0.01", "grpo_controls_20261017/evaluations/kl_1e-02/greedy/grpo"),
    "clip": ("GRPO, clipping 0.1", "grpo_controls_20261017/evaluations/clip_0p1/greedy/grpo"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def load_policy(directory, sources, expected=300):
    rows = {}
    configs = []
    for shard in range(3):
        path = directory / f"shard_{shard}.jsonl"
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        part = [json.loads(line) for line in raw.splitlines() if line.strip()]
        config_path = directory / f"shard_{shard}_config.json"
        config_raw = config_path.read_bytes()
        sources[str(config_path)] = hashlib.sha256(config_raw).hexdigest()
        config = json.loads(config_raw)
        if (len(part) != 100 or config["mode"] != "greedy" or config["temperature"] != 0
                or config["max_new_tokens"] != 3000 or config["num_samples"] != 1
                or config["top_p"] != 1 or config["top_k"] != 0):
            raise ValueError(f"Incomplete or incompatible greedy cohort: {path}")
        if [r["source_index"] for r in part] != config["source_indices"]:
            raise ValueError(f"Row order differs from saved configuration: {path}")
        if [r["prompt_hash"] for r in part] != config["prompt_hashes"]:
            raise ValueError(f"Prompt hash differs from saved configuration: {path}")
        configs.append(config)
        for row in part:
            if row["source_index"] in rows or len(row["candidates"]) != 1:
                raise ValueError(f"Duplicate problem or unexpected candidate count: {path}")
            candidate = row["candidates"][0]
            if not isinstance(candidate["correct"], bool) or not candidate.get("raw_solution"):
                raise ValueError(f"Missing saved solution or correctness: {path}")
            rows[row["source_index"]] = row
    if len(rows) != expected:
        raise ValueError("Incomplete development cohort")
    if any(c["model_path"] != configs[0]["model_path"] or c["stop_ids"] != configs[0]["stop_ids"] for c in configs):
        raise ValueError("Model or stopping conventions change between shards")
    return rows, configs[0]


def collect(root):
    policies, configs, sources = {}, {}, {}
    for name, (_, relative) in POLICIES.items():
        policies[name], configs[name] = load_policy(root / relative, sources)
    baseline = policies["base"]
    for name, rows in policies.items():
        if rows.keys() != baseline.keys():
            raise ValueError(f"Development IDs differ: {name}")
        if configs[name]["stop_ids"] != configs["base"]["stop_ids"]:
            raise ValueError(f"Stopping conventions differ: {name}")
        for key, row in rows.items():
            for field in ("problem", "prompt_hash", "reference_answer", "type", "level"):
                if row[field] != baseline[key][field]:
                    raise ValueError(f"Mismatched {field}: {name}/{key}")
    return policies, configs, sources


def candidate(rows, key):
    return rows[key]["candidates"][0]


def inventory(policies):
    names = list(POLICIES)
    print("POLICIES", " ".join(names))
    for key, row in policies["base"].items():
        flags = "".join("+" if candidate(policies[n], key)["correct"] else "-" for n in names)
        lengths = [candidate(policies[n], key)["generated_tokens"] for n in ("base", "grpo", "sft", "lr3")]
        print(key, flags, lengths, row["type"], row["level"], row["problem"].replace("\n", " "))


def validate_annotations(spec, policies):
    seen = set()
    for case in spec["cases"]:
        key = case["source_index"]
        if key in seen or key not in policies["base"]:
            raise ValueError("Duplicate or unknown annotated problem")
        seen.add(key)
        for name, expected in case["expected"].items():
            if candidate(policies[name], key)["correct"] != expected:
                raise ValueError(f"Annotated outcome differs from evidence: {key}/{name}")
        for evidence in case["evidence"]:
            raw = candidate(policies[evidence["policy"]], key)["raw_solution"]
            if not evidence["quote"] or evidence["quote"] not in raw:
                raise ValueError(f"Quoted evidence is not an exact source substring: {key}/{evidence['policy']}")
        if not set(case["expected"]) <= set(case["policies"]):
            raise ValueError("Expected policy missing from comparison")


def paired_metrics(rows, anchor):
    if rows.keys() != anchor.keys():
        raise ValueError("Cannot compare different problem sets")
    return {
        "n": len(rows),
        "correct": sum(candidate(rows, i)["correct"] for i in rows),
        "fixed": [i for i in rows if candidate(rows, i)["correct"] and not candidate(anchor, i)["correct"]],
        "broken": [i for i in rows if not candidate(rows, i)["correct"] and candidate(anchor, i)["correct"]],
        "truncated": sum(candidate(rows, i)["truncated"] for i in rows),
    }


def independent_checks():
    """Small reviewer-authored checks; never evaluate generated model code."""
    from collections import Counter
    from fractions import Fraction
    from itertools import permutations
    from math import comb, pi
    import sympy as sp

    x, y, k = sp.symbols("x y k", real=True)
    f = x / (x**2 - x + 1)
    quartic = x**4 + 6*x**2 + 25
    other = 3*x**4 + 4*x**2 + 28*x + 5
    factor = x**2 - 2*x + 5
    counts = Counter("".join(map(str, [*range(100, 126), *range(200, 226)])))
    checks = {
        "quadratic_vertex": sp.expand((x-5)**2 - 1) == x**2 - 10*x + 24,
        "three_digit_multiples": len([n for n in range(100, 1000) if n % 55 == 0]) == 17,
        "conjugate_base_and_exponents": sp.expand((sp.sqrt(2)-1)*(sp.sqrt(2)+1)) == 1 and sp.simplify(-(1-sp.sqrt(3))-(1+sp.sqrt(3))) == -2,
        "range_endpoints_and_discriminant": f.subs(x, -1) == -sp.Rational(1, 3) and f.subs(x, 1) == 1 and sp.expand((y+1)**2-4*y**2 + (y-1)*(3*y+1)) == 0,
        "negative_residue": -3737 % 8 == 7,
        "functional_equation": sp.simplify(3*((3-2*x**5)/(5*x**2)).subs(x, 1/x) + 2*((3-2*x**5)/(5*x**2))/x - x**2) == 0 and Fraction(67, 20) == Fraction(3, 20) + Fraction(64, 20),
        "distance_exactness": sp.sqrt(41) + sp.sqrt(41) == 2*sp.sqrt(41) and sp.Rational(1281, 100)**2 != 164,
        "digit_packages": counts["1"] == 52 and counts["2"] == 44 and max(counts.values()) == 52,
        "quartic_root_witnesses": [sum(m for root, m in sp.roots(p, x).items() if root.is_integer) for p in (x**4+1, x*(x**3-2), x**2*(x**2+1), x**4)] == [0, 1, 2, 4],
        "involution_nondegeneracy": sp.Matrix([[2, 3], [k, -2]])**2 == (4+3*k)*sp.eye(2) and sp.simplify((2*x+3)/(-sp.Rational(4, 3)*x-2)) == -sp.Rational(3, 2),
        "median_code_discrepancy": sp.Rational(8**2+8**2-5**2, 2) == sp.Rational(103, 2) and (8**2+8**2-10**2/2)/2 == 39,
        "distinct_digit_permutations": len(set(permutations([2, 3, 5, 5, 5, 6, 6], 3))) == 43,
        "divisor_minimum": next(n for n in range(1, 193) if sp.divisor_count(n) == 14) == 192 and sp.divisor_count(30) == 8,
        "polynomial_shift": sp.expand(2*(x-4)**2-(x-4)+7) == 2*x**2-17*x+43,
        "principal_angle": 0 < 7-2*pi < pi,
        "shared_quadratic_factor": sp.rem(quartic, factor, x) == 0 and sp.rem(other, factor, x) == 0 and sp.expand(3*quartic-other) == 14*factor,
        "silverware_probability": Fraction(6**3, comb(18, 3)) == Fraction(9, 34) and Fraction(9, 34) != Fraction(1, 4),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Independent mathematical check failed: {failed}")
    return {name: bool(passed) for name, passed in checks.items()}


def build_payload(spec, policies, configs, sources, references):
    validate_annotations(spec, policies)
    cases = []
    for annotation in spec["cases"]:
        key = annotation["source_index"]
        row = policies["base"][key]
        reference = references[key]
        if reference["problem"] != row["problem"]:
            raise ValueError("Reference solution belongs to a different problem")
        cases.append({**annotation, "problem": row["problem"], "type": row["type"], "level": row["level"],
                      "reference_answer": row["reference_answer"], "dataset_reference_solution": reference["solution"],
                      "responses": {name: candidate(rows, key) for name, rows in policies.items()}})
    anchors = {"format": "lr3", "length": "lr3", "step512": "lr3", "kl": "grpo", "clip": "grpo"}
    return {"selection": spec["selection"], "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "Same dev300 IDs/problem text/reference/prompt hashes/stops; greedy, top_p=1, top_k=0, cap3000. Historical correctness labels retained. No model generation or tool execution.",
            "cases": cases, "sources": sources, "policy_configs": configs,
            "full_dev300": {name: {"anchor": anchors.get(name, "base"), **paired_metrics(rows, policies[anchors.get(name, "base")])}
                            for name, rows in policies.items()}}


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


INTRO = (
    "These are 17 deliberately selected, retrospective case studies of Qwen2.5-Math-1.5B base and its adapted policies, from the fixed 300-problem MATH-train development split, disjoint from the 512 pilot training problems. They are not MATH-500 test cases or a random sample of errors. Selection covers improvements, regressions, persistent failures, grading-sensitive changes, and correct answers with flawed explanations. No case-frequency or selected-case accuracy is a population estimate.",
    "All 12 saved policy cohorts are complete on the same 300 problems. Before comparison, the exporter checks IDs, problem text, reference answers, prompt hashes, greedy settings, cap, and stop tokens across policies. These fields establish evaluation matching, not equality of all training conditions. Raw responses and source hashes are archived in scratch; short quotations below must exactly match them. Original correctness labels and extracted answers are retained, even when a case exposes a grading problem.",
    "Reference reasoning is an independently written mathematical explanation, not a quotation from the dataset solution. The original dataset solution is retained in the scratch evidence export. Diagnoses are qualitative mathematical readings, not a trained verifier or formal proof certificate. Findings describe observed outputs, not internal model reasoning or a causal explanation of training.",
    "Code, output blocks, timeout messages and diagrams inside a completion are model-generated text: the evaluation did not execute those blocks. Independent checks use separately authored arithmetic/symbolic checks, never execution of model code. A correct final answer can coexist with invalid intermediate steps; a wrong score can coexist with correct mathematics. Only explicitly discussed steps have been reviewed, not every line of every response.",
    "GRPO and Dr. GRPO refer to the original LR1e-6/128-update arms. The LR3e-6 correctness-only arm anchors the format/length and 128-to-512 duration comparisons. KL0.01 and clip0.1 anchor to original GRPO. Original SFT uses human solutions and common evaluation stops; follow-up SFT/RFT are separate supervised controls. RFT uses filtered self-generated targets and is not an otherwise identical optimizer-only comparison. No new inference, training, or benchmark rescoring is performed.",
)


def render(payload, destination):
    if len(payload["cases"]) != 17:
        raise ValueError("This reviewed report requires exactly 17 cases")
    destination.mkdir(parents=True, exist_ok=True)
    lines = ["# Problem-Level Reasoning Case Studies", "", *[paragraph + "\n" for paragraph in INTRO],
             "![Selected-problem outcome and response-length matrix](case_matrix.png)", "",
             "## Full Development Context (Not the Selected-Case Sample)", "",
             "These transition counts use all 300 problems. They are stored-grader outcomes, not frequencies of mathematical error types.", "",
             "| Policy | Comparison anchor | Correct / 300 | Fixed | Broken | Truncated |",
             "|---|---|---:|---:|---:|---:|"]
    tex = [r"\documentclass[10pt,letterpaper]{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern,amsmath,amssymb,graphicx,booktabs,array,longtable,xurl,titlesec}",
           r"\usepackage[margin=0.75in]{geometry}", r"\usepackage[hidelinks]{hyperref}", r"\setlength{\parindent}{0pt}",
           r"\setlength{\parskip}{0.5em}", r"\setlength{\tabcolsep}{3pt}", r"\setlength{\emergencystretch}{3em}",
           r"\titlespacing*{\subsection}{0pt}{0.7em}{0.3em}",
           r"\title{Problem-Level Reasoning Case Studies\\Baseline, Supervised Adaptation, RL, and Ablations}",
           r"\author{Tuan Minh Pham, Tri Vo, Yahor Lechanka\\Purdue University}", r"\date{Saved-output analysis}",
           r"\begin{document}\maketitle\section*{Scope and Reading Guide}", *[tex_escape(p) + "\n\n" for p in INTRO],
           r"\section*{Selected-Problem Matrix}", r"\includegraphics[width=\linewidth]{case_matrix.pdf}",
           r"\section*{Full Development Context}",
           r"Stored-grader transitions on all 300 problems, not error-type frequencies in the selected sample.",
           r"\begin{longtable}{p{0.30\linewidth}p{0.29\linewidth}rrrr}\toprule Policy & Anchor & Correct & Fixed & Broken & Truncated\\\midrule"]
    for name, m in payload["full_dev300"].items():
        label, anchor = POLICIES[name][0], POLICIES[m["anchor"]][0]
        values = [m["correct"], len(m["fixed"]), len(m["broken"]), m["truncated"]]
        lines.append(f"| {label} | {anchor} | " + " | ".join(map(str, values)) + " |")
        tex.append(tex_escape(label) + " & " + tex_escape(anchor) + " & " + " & ".join(map(str, values)) + r"\\")
    tex.append(r"\bottomrule\end{longtable}")
    for number, case in enumerate(payload["cases"], 1):
        title = f"Case {number:02d}: {case['title']}"
        lines += ["", "## " + title, "", f"**{case['type']}, {case['level']} | {case['mechanism']}**", "",
                  "### Problem", "", case["problem"], "", "### Reference Reasoning", "", case["reference_reasoning"], "",
                  "### Saved Outputs", "", "| Policy | Saved grade | Extracted answer | Tokens | Truncated |", "|---|---|---|---:|---|"]
        tex += [r"\clearpage\section*{" + tex_escape(title) + "}", tex_escape(f"{case['type']}, {case['level']} | {case['mechanism']}"),
                r"\subsection*{Problem}", case["problem"], r"\subsection*{Reference Reasoning}", case["reference_reasoning"],
                r"\subsection*{Saved Outputs}", r"\small\begin{longtable}{p{0.29\linewidth}p{0.10\linewidth}p{0.35\linewidth}rr}\toprule Policy & Grade & Extracted answer & Tokens & Cap hit\\\midrule"]
        for name in case["policies"]:
            c = case["responses"][name]
            values = [POLICIES[name][0], "correct" if c["correct"] else "wrong", str(c["answer"]), str(c["generated_tokens"]), "yes" if c["truncated"] else "no"]
            lines.append("| " + " | ".join(v.replace("|", r"\|").replace("\n", " ") for v in values) + " |")
            tex.append(" & ".join(tex_escape(v) for v in values) + r"\\")
        tex += [r"\bottomrule\end{longtable}\normalsize", r"\subsection*{What Changes, and What Fails}", case["analysis"],
                r"\subsection*{Interpretation}", case["interpretation"], r"\subsection*{Exact Saved-Response Excerpts}"]
        lines += ["", "### What Changes, and What Fails", "", case["analysis"], "", "### Interpretation", "", case["interpretation"], "", "### Exact Saved-Response Excerpts"]
        for evidence in case["evidence"]:
            label = POLICIES[evidence["policy"]][0]
            lines += ["", f"**{label}:**", "", "````text", evidence["quote"], "````"]
            tex += [r"\textbf{" + tex_escape(label) + "}", r"\begin{quote}\small\raggedright " + tex_escape(evidence["quote"]).replace("\n", r"\par ") + r"\end{quote}"]
    conclusions = [
        "Separate three axes: mathematical reasoning, committing/extracting an answer, and stopping behavior. The 17-count case improves in score without a mathematical correction; the irrational-power case contains a genuine correction.",
        "Improvements coexist with regressions. Higher LR repairs a sign error and residue output, but still loses a coefficient in the functional equation. Longer RL improves polynomial elimination while losing an exact probability answer.",
        "Neither supervised adaptation nor RL universally dominates individual problems. SFT preserves an exact distance more compactly; RFT can recover a correct coefficient calculation, yet also reaches a correct count through wrong intermediate arithmetic and fails to terminate.",
        "Final-answer reward is not trace verification. Generated code-output consistency, algebraic equivalence, exactness, domain checks and proof completeness are distinct axes for future audits. These examples motivate those checks without estimating their prevalence or changing historical benchmark totals.",
    ]
    lines += ["", "## Cross-Case Findings", "", *["- " + c for c in conclusions]]
    tex += [r"\clearpage\section*{Cross-Case Findings}", *[tex_escape(c) + "\n\n" for c in conclusions], r"\end{document}"]
    (destination / "analysis.md").write_text("\n".join(lines) + "\n")
    (destination / "main.tex").write_text("\n".join(tex) + "\n")
    public = {k: v for k, v in payload.items() if k not in ("sources", "policy_configs", "cases")}
    public["cases"] = [{k: v for k, v in case.items() if k not in ("responses", "dataset_reference_solution")} for case in payload["cases"]]
    (destination / "case_summaries.json").write_text(json.dumps(public, indent=2, ensure_ascii=True) + "\n")


def plot_matrix(payload, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir() / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    cases, names = payload["cases"], list(POLICIES)
    scores = np.array([[c["responses"][n]["correct"] for n in names] for c in cases])
    tokens = np.array([[c["responses"][n]["generated_tokens"] for n in names] for c in cases])
    labels = [f"{i+1:02d}  {c['type']}" for i, c in enumerate(cases)]
    short = ["Base", "GRPO", "Dr.GRPO", "SFT", "SFT-new", "RFT", "LR3", "512", "Format", "Length", "KL.01", "Clip.1"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), layout="constrained")
    for ax in axes:
        ax.set_xticks(range(len(names)), short, rotation=55, ha="right")
        ax.set_yticks(range(len(cases)), labels)
        ax.tick_params(labelsize=9)
    axes[0].imshow(scores, cmap=ListedColormap(["#c64e52", "#2a987c"]), vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Saved-grader outcome (not proof validity)")
    for i, case in enumerate(cases):
        for j, name in enumerate(names):
            value = "1" if scores[i, j] else "0"
            if case["responses"][name]["truncated"]:
                value += "*"
            axes[0].text(j, i, value, ha="center", va="center", color="white", fontsize=8)
    im = axes[1].imshow(tokens, cmap="cividis", vmin=0, vmax=3000, aspect="auto")
    axes[1].set_title("Generated tokens, same 3,000-token cap")
    axes[1].tick_params(labelleft=False)
    fig.colorbar(im, ax=axes[1], label="Generated tokens", shrink=.75)
    fig.suptitle("Purposively selected development cases; 1 = stored correct, 0 = stored wrong, * = truncated", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"case_matrix.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, default=output_dir("rl_1p5b_study"))
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--inspect", type=int, nargs="+")
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES))
    parser.add_argument("--inspect_chars", type=int, default=4000, help="Display cap per response; full saved text is never modified")
    parser.add_argument("--annotations", type=Path, default=project_root() / "docs/problem_case_studies/annotations.json")
    parser.add_argument("--output_dir", type=Path, default=project_root() / "docs/problem_case_studies")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Plot the existing scratch export; no raw-policy reload")
    parser.add_argument("--archive", type=Path, default=output_dir("problem_case_studies/evidence.json"))
    args = parser.parse_args()
    archive = ensure_scratch_path(args.archive)
    if args.plot:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_matrix(read_json(archive), args.output_dir)
        return
    policies, configs, sources = collect(ensure_scratch_path(args.study_root))
    if args.build:
        from datasets import load_from_disk
        split = data_dir("rl_1p5b_pilot_v1/development")
        references = {r["source_index"]: r for r in load_from_disk(str(split))}
        if references.keys() != policies["base"].keys():
            raise ValueError("Saved cohorts are not the fixed development set")
        train = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/train")))
        if set(train["source_index"]) & references.keys():
            raise ValueError("Development cases overlap pilot training IDs")
        original = load_from_disk(str(data_dir("MATH")))["train"]
        for key, frozen in references.items():
            source = original[key]
            if source["problem"] != frozen["problem"]:
                raise ValueError("Original MATH source index no longer matches frozen problem")
            frozen["solution"] = source["solution"]
        for source in original.cache_files:
            path = Path(source["filename"])
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(split.iterdir()):
            if path.is_file():
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[str(args.annotations)] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
        source_path = Path(__file__).resolve()
        sources[str(source_path)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        payload = build_payload(read_json(args.annotations), policies, configs, sources, references)
        payload["independent_math_checks"] = independent_checks()
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
        render(payload, args.output_dir)
        print(json.dumps({"cases": len(payload["cases"]), "complete_policies": len(policies), "matched_problems": 300, "output": str(args.output_dir)}))
        return
    if args.inventory:
        inventory(policies)
    for key in args.inspect or []:
        row = policies["base"][key]
        print("PROBLEM", key, row["problem"], "REFERENCE", row["reference_answer"])
        for name in args.policies:
            c = candidate(policies[name], key)
            print("\nPOLICY", name, "CORRECT", c["correct"], "ANSWER", c["answer"], "TOKENS", c["generated_tokens"], "TRUNCATED", c["truncated"])
            raw = c["raw_solution"]
            print(raw if len(raw) <= args.inspect_chars else raw[:args.inspect_chars] + "\n[DISPLAY EXCERPT; remainder not printed]")


if __name__ == "__main__":
    main()
