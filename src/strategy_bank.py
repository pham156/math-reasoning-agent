from __future__ import annotations


STRATEGY_BANK: dict[str, str] = {
    "modular_arithmetic": "Use congruences, parity, residues, or divisibility constraints to reduce possible cases.",
    "factorization": "Rewrite expressions as products, exploit roots, or use identities such as difference of squares.",
    "casework": "Partition the problem into exhaustive cases where each case has simpler structure.",
    "invariant_or_monovariant": "Find a quantity that stays fixed or moves in one direction under allowed operations.",
    "extremal_argument": "Choose an object with maximal or minimal value and derive forced structure.",
    "induction": "Prove a base case and show the statement propagates from n to n+1 or a stronger form.",
    "bijection_or_counting": "Count the same set in two ways or construct a one-to-one correspondence.",
    "pigeonhole": "Compare objects to containers to force repetition or concentration.",
    "similar_triangles": "Use equal angles and proportional sides to transfer metric information.",
    "coordinate_or_vector": "Place the configuration in coordinates or vectors to turn geometry into algebra.",
    "bounding_inequality": "Use standard inequalities or bounding to trap the target expression.",
    "construction": "Build an example or witness satisfying the required conditions.",
    "contradiction": "Assume the negation and derive an impossibility.",
}


THEOREM_BANK: dict[str, str] = {
    "euclidean_algorithm": "GCD structure can be exposed by repeated division, Bezout relations, or coprimality.",
    "fermats_little_theorem": "For prime p and gcd(a,p)=1, a^(p-1) == 1 mod p.",
    "chinese_remainder_theorem": "Independent congruences modulo coprime moduli can be combined uniquely modulo their product.",
    "vieta_relations": "For polynomial roots, use sums and products of roots from the coefficients.",
    "am_gm": "The arithmetic mean is at least the geometric mean for nonnegative terms.",
    "cauchy_schwarz": "Relate sums of squares and products; useful for bounding expressions.",
    "pythagorean_theorem": "In right triangles, a^2 + b^2 = c^2.",
    "power_of_a_point": "Products of secant/tangent segment lengths from a point to a circle are equal.",
    "stars_and_bars": "Nonnegative integer solutions to x1+...+xk=n correspond to placing bars among stars.",
    "binomial_theorem": "Expand powers of sums and reason about binomial coefficients.",
    "bayes_or_conditional_probability": "Track conditional probabilities carefully by conditioning on the given event.",
}


def describe_patterns(strategies: list[str], theorem_patterns: list[str]) -> list[str]:
    lines = []
    for strategy in strategies:
        if strategy in STRATEGY_BANK:
            lines.append(f"- strategy/{strategy}: {STRATEGY_BANK[strategy]}")
    for theorem in theorem_patterns:
        if theorem in THEOREM_BANK:
            lines.append(f"- theorem/{theorem}: {THEOREM_BANK[theorem]}")
    return lines
