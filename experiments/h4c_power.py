"""Exact directional McNemar power helper for H4-C sample-size planning.

The paired primary endpoint depends on discordant outcomes. Planning assumptions are
therefore expressed as:
- p10: P(C1 SAER=1, C2 SAER=0)
- p01: P(C1 SAER=0, C2 SAER=1)

This helper is for pre-confirmatory planning only. Assumptions should be frozen from an
independent development split or an externally justified design target before collecting
confirmatory data.
"""
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache


@lru_cache(maxsize=None)
def _binomial_distribution(n: int, p: float) -> tuple[float, ...]:
    """Return a normalized binomial PMF without large integer-to-float products."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    if p == 0.0:
        return (1.0,) + (0.0,) * n
    if p == 1.0:
        return (0.0,) * n + (1.0,)

    mode = min(n, int(math.floor((n + 1) * p)))
    log_mode = (
        math.lgamma(n + 1)
        - math.lgamma(mode + 1)
        - math.lgamma(n - mode + 1)
        + mode * math.log(p)
        + (n - mode) * math.log1p(-p)
    )

    probabilities = [0.0] * (n + 1)
    probabilities[mode] = math.exp(log_mode)

    odds = p / (1.0 - p)
    for k in range(mode, n):
        probabilities[k + 1] = (
            probabilities[k]
            * (n - k)
            / (k + 1)
            * odds
        )

    inverse_odds = (1.0 - p) / p
    for k in range(mode, 0, -1):
        probabilities[k - 1] = (
            probabilities[k]
            * k
            / (n - k + 1)
            * inverse_odds
        )

    total = math.fsum(probabilities)
    if not math.isfinite(total) or total <= 0.0:
        raise ArithmeticError("failed to construct stable binomial distribution")
    return tuple(value / total for value in probabilities)


def _binomial_pmf(k: int, n: int, p: float) -> float:
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    return _binomial_distribution(n, p)[k]


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    return math.fsum(_binomial_distribution(n, p)[k:])


@lru_cache(maxsize=None)
def _critical_c1_only(discordant: int, alpha: float) -> int | None:
    if discordant < 0:
        raise ValueError("discordant must be nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    probabilities = _binomial_distribution(discordant, 0.5)
    tail = 0.0
    critical: int | None = None
    for c1_only in range(discordant, -1, -1):
        tail += probabilities[c1_only]
        if tail <= alpha:
            critical = c1_only
        else:
            break
    return critical


@lru_cache(maxsize=None)
def _conditional_rejection_probability(
    discordant: int,
    *,
    conditional_c1_probability: float,
    alpha: float,
) -> float:
    critical = _critical_c1_only(discordant, alpha)
    if critical is None:
        return 0.0
    return binomial_upper_tail(
        critical,
        discordant,
        conditional_c1_probability,
    )


def exact_directional_mcnemar_power(
    n_pairs: int,
    *,
    p10: float,
    p01: float,
    alpha: float = 0.05,
) -> float:
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    if not 0.0 <= p10 <= 1.0 or not 0.0 <= p01 <= 1.0:
        raise ValueError("p10 and p01 must be between 0 and 1")
    if p10 + p01 > 1.0:
        raise ValueError("p10 + p01 must not exceed 1")
    if p10 <= p01:
        raise ValueError("directional H4-C planning requires p10 > p01")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    discordance_probability = p10 + p01
    conditional_c1_probability = p10 / discordance_probability

    discordant_distribution = _binomial_distribution(
        n_pairs,
        discordance_probability,
    )
    contributions = []
    for discordant, p_discordant_count in enumerate(discordant_distribution):
        conditional_rejection = _conditional_rejection_probability(
            discordant,
            conditional_c1_probability=conditional_c1_probability,
            alpha=alpha,
        )
        contributions.append(p_discordant_count * conditional_rejection)
    return math.fsum(contributions)


def minimum_pairs_for_power(
    *,
    p10: float,
    p01: float,
    target_power: float = 0.90,
    alpha: float = 0.05,
    max_pairs: int = 2_000,
) -> tuple[int, float]:
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be between 0 and 1")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")

    for n_pairs in range(1, max_pairs + 1):
        power = exact_directional_mcnemar_power(
            n_pairs,
            p10=p10,
            p01=p01,
            alpha=alpha,
        )
        if power >= target_power:
            return n_pairs, power
    raise ValueError("target power not reached before max_pairs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan H4-C paired sample size for a directional exact McNemar test"
    )
    parser.add_argument("--p10", type=float, required=True)
    parser.add_argument("--p01", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.90)
    parser.add_argument("--max-pairs", type=int, default=2_000)
    args = parser.parse_args()

    n_pairs, achieved_power = minimum_pairs_for_power(
        p10=args.p10,
        p01=args.p01,
        target_power=args.target_power,
        alpha=args.alpha,
        max_pairs=args.max_pairs,
    )
    output = {
        "planning_scope": (
            "Pre-confirmatory sample-size planning only; freeze assumptions before "
            "collecting confirmatory H4-C data."
        ),
        "alternative": "C1 SAER > C2 SAER",
        "alpha_one_sided": args.alpha,
        "target_power": args.target_power,
        "p10_c1_success_c2_failure": args.p10,
        "p01_c1_failure_c2_success": args.p01,
        "minimum_pairs": n_pairs,
        "achieved_power": achieved_power,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
