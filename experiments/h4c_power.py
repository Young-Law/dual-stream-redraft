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


def _binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def _binomial_upper_tail(k: int, n: int, p: float) -> float:
    return sum(_binomial_pmf(j, n, p) for j in range(k, n + 1))


@lru_cache(maxsize=None)
def _critical_c1_only(discordant: int, alpha: float) -> int | None:
    for c1_only in range(discordant + 1):
        if _binomial_upper_tail(c1_only, discordant, 0.5) <= alpha:
            return c1_only
    return None


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
    return _binomial_upper_tail(
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

    power = 0.0
    for discordant in range(n_pairs + 1):
        p_discordant_count = _binomial_pmf(
            discordant,
            n_pairs,
            discordance_probability,
        )
        conditional_rejection = _conditional_rejection_probability(
            discordant,
            conditional_c1_probability=conditional_c1_probability,
            alpha=alpha,
        )
        power += p_discordant_count * conditional_rejection
    return power


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
