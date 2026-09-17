from __future__ import annotations

from experiments.h4_hybrid_widening import run_benchmark


def test_h4_top3_concealment_hybrid_adds_nonzero_coverage() -> None:
    report = run_benchmark(trials=10, sequence_length=2_000, event_count=10, rate_ppm=5_000)
    scenario = report["scenarios"]["top3_concealment"]
    short = scenario["10"]
    long = scenario["100"]

    assert short["rank_only"]["event_recall_mean"] == 0.0
    assert long["rank_only"]["event_recall_mean"] == 0.0
    assert short["hybrid"]["event_recall_mean"] > 0.0
    assert long["hybrid"]["event_recall_mean"] >= short["hybrid"]["event_recall_mean"]


def test_h4_budget_matched_random_is_sanity_baseline_not_semantic_control() -> None:
    report = run_benchmark(trials=20, sequence_length=2_000, event_count=10, rate_ppm=5_000)
    scenario = report["scenarios"]["top3_concealment"]["100"]

    hybrid_rate = scenario["hybrid"]["widen_rate_mean"]
    random_rate = scenario["budget_matched_random"]["widen_rate_mean"]
    assert hybrid_rate == random_rate

    # With no rank-trigger information, keyed selection should behave like a
    # uniformly distributed budget-matched sampling control at the mechanism level.
    assert abs(
        scenario["hybrid"]["event_recall_mean"]
        - scenario["budget_matched_random"]["event_recall_mean"]
    ) < 0.20


def test_h4_mixed_attack_hybrid_never_loses_rank_trigger_coverage() -> None:
    report = run_benchmark(trials=10, sequence_length=2_000, event_count=10, rate_ppm=5_000)
    scenario = report["scenarios"]["mixed_25pct_rank_triggered"]

    for policies in scenario.values():
        assert policies["hybrid"]["event_recall_mean"] >= policies["rank_only"]["event_recall_mean"]
