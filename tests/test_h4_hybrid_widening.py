from experiments.h4_hybrid_widening import _trial, run_benchmark


def test_top3_concealment_rank_only_has_no_coverage():
    result = _trial(
        sequence_length=2_000,
        event_length=20,
        event_count=10,
        rate_ppm=50_000,
        seed=1,
        rank_trigger_fraction=0.0,
    )
    assert result["rank_only"]["event_recall"] == 0.0
    assert result["hybrid_keyed"]["event_recall"] > 0.0


def test_hybrid_never_discards_rank_triggered_coverage():
    for seed in range(1, 21):
        result = _trial(
            sequence_length=2_000,
            event_length=10,
            event_count=10,
            rate_ppm=5_000,
            seed=seed,
            rank_trigger_fraction=0.5,
        )
        assert result["hybrid_keyed"]["event_recall"] >= result["rank_only"]["event_recall"]
        assert result["rank_plus_uniform"]["event_recall"] >= result["rank_only"]["event_recall"]
        assert result["hybrid_keyed"]["widen_rate"] == result["rank_plus_uniform"]["widen_rate"]


def test_report_labels_scope_as_synthetic_mechanism_validation():
    report = run_benchmark(trials=2, sequence_length=2_000, event_count=4, rate_ppm=5_000)
    interpretation = str(report["interpretation"])
    assert "Synthetic mechanism validation only" in interpretation
    assert "does not measure deceptive-alignment detection" in interpretation
    assert "rank_plus_uniform" in str(report["control"])
