from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternRule:
    name: str
    keywords: tuple[str, ...]


TOPIC_RULES: tuple[PatternRule, ...] = (
    PatternRule("number_theory", ("integer", "prime", "divisible", "mod", "congru", "gcd", "lcm", "remainder", "diophantine")),
    PatternRule("algebra", ("polynomial", "quadratic", "equation", "root", "factor", "expression", "symmetric", "inequality")),
    PatternRule("geometry", ("triangle", "circle", "angle", "parallel", "perpendicular", "radius", "area", "length", "similar")),
    PatternRule("combinatorics", ("count", "ways", "arrange", "choose", "permutation", "combination", "graph", "color", "pigeonhole")),
    PatternRule("probability", ("probability", "random", "expected", "expectation", "variance", "dice", "coin")),
    PatternRule("precalculus", ("function", "trigonometric", "sin", "cos", "tan", "log", "exponential", "sequence", "series")),
)

STRATEGY_RULES: tuple[PatternRule, ...] = (
    PatternRule("modular_arithmetic", ("mod", "congru", "remainder", "divisible", "parity", "even", "odd")),
    PatternRule("factorization", ("factor", "product", "difference of squares", "polynomial", "root")),
    PatternRule("casework", ("case", "cases", "without loss", "wlog", "split")),
    PatternRule("invariant_or_monovariant", ("invariant", "unchanged", "monovariant", "always increases", "always decreases")),
    PatternRule("extremal_argument", ("maximum", "minimum", "largest", "smallest", "extreme")),
    PatternRule("induction", ("induction", "base case", "inductive")),
    PatternRule("bijection_or_counting", ("bijection", "count", "choose", "arrange", "permutation", "combination")),
    PatternRule("pigeonhole", ("pigeonhole", "at least", "at most", "boxes", "holes")),
    PatternRule("similar_triangles", ("similar triangle", "similarity", "angle chasing", "parallel")),
    PatternRule("coordinate_or_vector", ("coordinate", "vector", "slope", "distance formula")),
    PatternRule("bounding_inequality", ("inequality", "am-gm", "cauchy", "jensen", "bound")),
    PatternRule("construction", ("construct", "exists", "there exists", "example")),
    PatternRule("contradiction", ("contradiction", "assume", "suppose not", "impossible")),
)

THEOREM_RULES: tuple[PatternRule, ...] = (
    PatternRule("euclidean_algorithm", ("gcd", "euclidean", "bezout")),
    PatternRule("fermats_little_theorem", ("fermat", "prime p", "mod p")),
    PatternRule("chinese_remainder_theorem", ("chinese remainder", "crt", "relatively prime moduli")),
    PatternRule("vieta_relations", ("roots", "sum of roots", "product of roots", "vieta")),
    PatternRule("am_gm", ("am-gm", "arithmetic mean", "geometric mean")),
    PatternRule("cauchy_schwarz", ("cauchy", "cauchy-schwarz", "cs inequality")),
    PatternRule("pythagorean_theorem", ("right triangle", "hypotenuse", "pythagorean")),
    PatternRule("power_of_a_point", ("power of a point", "tangent", "secant")),
    PatternRule("stars_and_bars", ("stars and bars", "nonnegative integer solutions")),
    PatternRule("binomial_theorem", ("binomial", "coefficient", "pascal")),
    PatternRule("bayes_or_conditional_probability", ("conditional probability", "given that", "bayes")),
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _matches(rule: PatternRule, text: str) -> bool:
    return any(keyword in text for keyword in rule.keywords)


def infer_labels(problem: str, solution: str = "", dataset_type: str = "") -> dict:
    text = normalize_text(f"{dataset_type} {problem} {solution}")
    topics = [rule.name for rule in TOPIC_RULES if _matches(rule, text)]
    strategies = [rule.name for rule in STRATEGY_RULES if _matches(rule, text)]
    theorems = [rule.name for rule in THEOREM_RULES if _matches(rule, text)]

    if dataset_type:
        normalized_type = normalize_text(dataset_type).replace(" ", "_")
        if normalized_type and normalized_type not in topics:
            topics.insert(0, normalized_type)

    return {
        "topics": topics[:4] or ["unknown"],
        "strategies": strategies[:6],
        "theorem_patterns": theorems[:6],
    }


def enrich_example(example: dict) -> dict:
    labels = infer_labels(
        problem=example.get("problem", ""),
        solution=example.get("solution", ""),
        dataset_type=example.get("type", example.get("category", "")),
    )
    enriched = dict(example)
    enriched.update(labels)
    return enriched


def labels_for_query(problem: str) -> dict:
    return infer_labels(problem=problem)
