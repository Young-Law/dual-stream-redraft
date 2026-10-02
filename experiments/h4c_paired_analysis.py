"""Paired H4-C C1/C2 analysis for development and frozen result artifacts.

The primary comparison is within matched C1/C2 pairs. This module reports raw outcome
counts, SAER rates, the paired rate difference, a paired normal-approximation confidence
interval, and an exact one-sided McNemar/binomial test based on discordant pairs.

When the source report is labeled non-confirmatory, every statistic emitted here remains
non-confirmatory as well.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


_Z_975 = 1.959963984540054


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _binomial_upper_tail(k: int, n: int, p: float = 0.5) -> float:
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    return sum(
        math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j))
        for j in range(k, n + 1)
    )


def _paired_difference_ci(values: list[int]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one pair is required")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return -1.0, 1.0
    sample_variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    se = math.sqrt(sample_variance / n)
    return (
        max(-1.0, mean - _Z_975 * se),
        min(1.0, mean + _Z_975 * se),
    )


def _pair_key(record: dict) -> str:
    digest = record.get("public_replay_context_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("record is missing public_replay_context_digest")
    return digest


def analyze_report(report: dict) -> dict:
    rows = report.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report records must be a nonempty list")

    pairs: dict[str, dict[str, dict]] = {}
    for record in rows:
        if not isinstance(record, dict):
            raise ValueError("every record must be an object")
        spec = record.get("spec")
        if not isinstance(spec, dict):
            raise ValueError("record is missing spec")
        condition = spec.get("condition")
        if condition not in {"C1", "C2"}:
            continue
        key = _pair_key(record)
        bucket = pairs.setdefault(key, {})
        if condition in bucket:
            raise ValueError(f"duplicate {condition} record for pair {key}")
        bucket[condition] = record

    if not pairs:
        raise ValueError("report contains no C1/C2 pairs")

    incomplete = [key for key, pair in pairs.items() if set(pair) != {"C1", "C2"}]
    if incomplete:
        raise ValueError(
            "incomplete C1/C2 pair(s): " + ", ".join(sorted(incomplete))
        )

    both = c1_only = c2_only = neither = 0
    c1_task_success = c2_task_success = 0
    c1_exposed_successes = c2_exposed_successes = 0
    differences: list[int] = []
    family_counts: dict[str, dict[str, int]] = {}

    for key, pair in pairs.items():
        c1 = pair["C1"]
        c2 = pair["C2"]
        if c1.get("public_replay_context_digest") != c2.get(
            "public_replay_context_digest"
        ):
            raise ValueError(f"public replay context mismatch for pair {key}")

        c1_result = c1.get("result")
        c2_result = c2.get("result")
        if not isinstance(c1_result, dict) or not isinstance(c2_result, dict):
            raise ValueError(f"missing result object for pair {key}")

        c1_saer = c1_result.get("successful_adaptive_evasion")
        c2_saer = c2_result.get("successful_adaptive_evasion")
        if not _is_bool(c1_saer) or not _is_bool(c2_saer):
            raise ValueError(f"C1/C2 SAER must be boolean for pair {key}")

        c1_task = c1_result.get("task_success")
        c2_task = c2_result.get("task_success")
        c1_exposed = c1_result.get("audit_exposed")
        c2_exposed = c2_result.get("audit_exposed")
        if not all(_is_bool(value) for value in (c1_task, c2_task, c1_exposed, c2_exposed)):
            raise ValueError(f"task/audit fields must be boolean for pair {key}")

        c1_task_success += int(c1_task)
        c2_task_success += int(c2_task)
        c1_exposed_successes += int(c1_task and c1_exposed)
        c2_exposed_successes += int(c2_task and c2_exposed)

        c1_value = int(c1_saer)
        c2_value = int(c2_saer)
        differences.append(c1_value - c2_value)

        if c1_saer and c2_saer:
            both += 1
        elif c1_saer:
            c1_only += 1
        elif c2_saer:
            c2_only += 1
        else:
            neither += 1

        family = str(c1.get("scenario_family", "unspecified"))
        family_bucket = family_counts.setdefault(
            family,
            {"pairs": 0, "c1_saer": 0, "c2_saer": 0},
        )
        family_bucket["pairs"] += 1
        family_bucket["c1_saer"] += c1_value
        family_bucket["c2_saer"] += c2_value

    n = len(pairs)
    c1_total = both + c1_only
    c2_total = both + c2_only
    difference = (c1_total - c2_total) / n
    ci_low, ci_high = _paired_difference_ci(differences)
    discordant = c1_only + c2_only
    exact_one_sided_p = (
        _binomial_upper_tail(c1_only, discordant, 0.5)
        if discordant
        else 1.0
    )

    family_summary = {}
    for family, counts in sorted(family_counts.items()):
        family_n = counts["pairs"]
        family_summary[family] = {
            **counts,
            "c1_saer_rate": counts["c1_saer"] / family_n,
            "c2_saer_rate": counts["c2_saer"] / family_n,
            "difference": (
                counts["c1_saer"] - counts["c2_saer"]
            ) / family_n,
        }

    source_scope = str(report.get("scope", ""))
    non_confirmatory = "non-confirmatory" in source_scope.casefold()
    return {
        "benchmark_id": report.get("benchmark_id"),
        "source_scope": source_scope,
        "analysis_scope": (
            "Non-confirmatory paired development analysis."
            if non_confirmatory
            else "Paired H4-C analysis; interpret according to the frozen study status."
        ),
        "number_of_pairs": n,
        "saer": {
            "c1_count": c1_total,
            "c2_count": c2_total,
            "c1_rate": c1_total / n,
            "c2_rate": c2_total / n,
            "paired_difference_c1_minus_c2": difference,
            "paired_difference_95pct_ci": [ci_low, ci_high],
            "ci_method": "paired normal approximation over per-pair differences",
        },
        "discordant_pairs": {
            "c1_success_c2_failure": c1_only,
            "c1_failure_c2_success": c2_only,
            "discordant_total": discordant,
            "exact_one_sided_mcnemar_p": exact_one_sided_p,
            "alternative": "C1 SAER > C2 SAER",
        },
        "joint_outcomes": {
            "both_success": both,
            "c1_only": c1_only,
            "c2_only": c2_only,
            "neither_success": neither,
        },
        "task_success": {
            "c1_count": c1_task_success,
            "c2_count": c2_task_success,
            "c1_rate": c1_task_success / n,
            "c2_rate": c2_task_success / n,
        },
        "audit_exposure_conditional_counts": {
            "c1_exposed_and_task_success": c1_exposed_successes,
            "c2_exposed_and_task_success": c2_exposed_successes,
        },
        "scenario_family": family_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze paired H4-C C1/C2 results")
    parser.add_argument("report", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    analysis = analyze_report(report)
    rendered = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
