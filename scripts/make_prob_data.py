"""Generate small probability word problems with exact fraction answers.

Writes a training set and a disjoint held-out eval set (no shared prompts). The eval set
also includes the three hand-written prompts from prompts/probability.txt.

Usage: python scripts/make_prob_data.py [--out-dir data] [--n-train 100] [--n-eval 30] [--seed 0]
"""

import argparse
import json
import random
from fractions import Fraction
from math import comb
from pathlib import Path

SUFFIX = " Give the final answer as a simplified fraction in \\boxed{}."
COUNT = {2: "twice", 3: "3 times", 4: "4 times", 5: "5 times", 6: "6 times", 7: "7 times", 8: "8 times"}


def at_least_heads(rng):
    n = rng.randint(3, 8)
    k = rng.randint(1, n - 1)
    p = Fraction(sum(comb(n, i) for i in range(k, n + 1)), 2**n)
    return f"A fair coin is flipped {COUNT[n]}. What is the probability of getting at least {k} heads?", p


def expected_face(rng):
    n, face = rng.randint(2, 12), rng.randint(1, 6)
    return f"A fair six-sided die is rolled {n} times. What is the expected number of {face}s seen?", Fraction(n, 6)


def bayes_coin(rng):
    p = Fraction(rng.choice([55, 6, 65, 7, 75, 8, 85, 9]), 1)
    p = p / (100 if p > 10 else 10)
    side = rng.choice(["heads", "tails"])
    like = p if side == "heads" else 1 - p
    posterior = like / (like + Fraction(1, 2))
    return (
        f"There are two coins: a loaded coin that lands heads with probability {float(p):g}, and a fair coin. "
        f"One coin is selected uniformly at random and flipped, and it lands {side}. "
        f"What is the probability that the loaded coin was selected?",
        posterior,
    )


def dice_sum(rng):
    s = rng.randint(2, 12)
    ways = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == s)
    return f"Two fair six-sided dice are rolled. What is the probability that their sum is {s}?", Fraction(ways, 36)


def urn_draw(rng):
    r, b = rng.randint(2, 7), rng.randint(2, 7)
    p = Fraction(r, r + b) * Fraction(r - 1, r + b - 1)
    return (
        f"An urn has {r} red balls and {b} blue balls. Two balls are drawn without replacement. "
        f"What is the probability that both are red?",
        p,
    )


FAMILIES = [at_least_heads, expected_face, bayes_coin, dice_sum, urn_draw]
HANDWRITTEN = [
    ("A fair coin is flipped 4 times. What is the probability of getting at least 2 heads?", Fraction(11, 16)),
    ("A fair six-sided die is rolled 5 times. What is the expected number of 4s seen?", Fraction(5, 6)),
    (
        "There are two coins: a loaded coin that lands heads with probability 0.6, and a fair coin. One coin is "
        "selected uniformly at random and flipped, and it lands heads. What is the probability that the loaded "
        "coin was selected?",
        Fraction(6, 11),
    ),
]


def fmt(p):
    return str(p.numerator) if p.denominator == 1 else f"{p.numerator}/{p.denominator}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reserved = {q for q, _ in HANDWRITTEN}
    eval_rows = [{"prompt": q, "answer": fmt(a), "family": "handwritten"} for q, a in HANDWRITTEN]
    train_rows, seen = [], set(reserved)
    # Fill eval first so train never contains an eval prompt. Families have few unique
    # problems (dice_sum has 11), so pick them at random and skip duplicates.
    for rows, n in [(eval_rows, args.n_eval + len(HANDWRITTEN)), (train_rows, args.n_train)]:
        attempts = 0
        while len(rows) < n:
            family = rng.choice(FAMILIES)
            q, a = family(rng)
            attempts += 1
            if q in seen:
                if attempts > 100 * n:
                    raise SystemExit(f"Could not generate {n} unique problems")
                continue
            seen.add(q)
            rows.append({"prompt": q + SUFFIX, "answer": fmt(a), "family": family.__name__})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("prob_train.jsonl", train_rows), ("prob_eval.jsonl", eval_rows)]:
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"Wrote {len(rows)} rows to {out / name}")


if __name__ == "__main__":
    main()
