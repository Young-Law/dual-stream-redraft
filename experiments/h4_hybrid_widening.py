from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from dualstream.compact_evidence import (
    ADAPTIVE_POLICY_HYBRID,
    keyed_sample_selected,
)

DEFAULT_SEQUENCE_LENGTH = 10_000
DEFAULT_EVENT_COUNT = 20
DEFAULT_RATE_PPM = 5_000
DEFAULT_TRIALS = 100
DEFAULT_LENGTHS = (1, 10, 50, 100, 500)
BASE_K = 3
MAX_K = 10
POLICY_VERSION = 2101
AUDIT_KEY_ID = 7
PROFILE_ID = "DSA-CI-Lite"
BENCHMARK_ID = "H4-SYNTHETIC-CONCEALMENT-v1"


@dataclass(frozen=True)
class Event:
    start: int
    length: int
    rank_triggered: bool


def _event_starts(sequence_length: int, event_length: int, event_count: int, seed: int) -> list[int]:
    if event_count < 1:
        raise ValueError("event_count must be positive")
    if event_length < 1 or event_length > sequence_length:
        raise ValueError("event_length must fit inside the sequence")
    bucket = sequence_length // event_count
    if bucket < event_length:
        raise ValueError("event_length is too large for non-overlapping bucket placement")
    rng = random.Random(seed)
    starts: list[int] = []
    for index in range(event_count):
        lo = index * bucket
        hi = min((index + 1) * bucket - event_length, sequence_length - event_length)
        starts.append(rng.randint(lo, hi))
    return starts


def _trial(
    *,
    sequence_length: int,
    event_length: int,
    event_count: int,
    rate_ppm: int,
    seed: int,
    rank_trigger_fraction: float,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    starts = _event_starts(sequence_length, event_length, event_count, seed + 999)
    events: list[Event] = []
    attack_tokens: set[int] = set()
    rank_triggered_tokens: set[int] = set()

    for start in starts:
        rank_triggered = rng.random() < rank_trigger_fraction
        event = Event(start, event_length, rank_triggered)
        events.append(event)
        tokens = set(range(start, start + event_length))
        attack_tokens.update(tokens)
        if rank_triggered:
            rank_triggered_tokens.update(tokens)

    commit_identity = hashlib.sha256(
        f"h4:{seed}:{sequence_length}:{event_length}:{event_count}:{rank_trigger_fraction}".encode()
    ).hexdigest()
    audit_key = hashlib.sha256(f"h4-key:{seed}".encode()).digest()

    rank_only_widened = set(rank_triggered_tokens)
    hybrid_widened = set(rank_triggered_tokens)

    for token_index in range(sequence_length):
        # The v2.10.1 encoder samples only otherwise-untriggered tokens. Rank-triggered
        # positions are already widened and therefore do not consume stochastic selections.
        if token_index in rank_triggered_tokens:
            continue
        if keyed_sample_selected(
            audit_key,
            commit_identity=commit_identity,
            sequence_id=seed,
            token_index=token_index,
            policy_version=POLICY_VERSION,
            rate_ppm=rate_ppm,
            benchmark_id=BENCHMARK_ID,
            audit_key_id=AUDIT_KEY_ID,
            profile_id=PROFILE_ID,
            base_k=BASE_K,
            max_adaptive_k=MAX_K,
            adaptive_policy=ADAPTIVE_POLICY_HYBRID,
            canary_eval=False,
        ):
            hybrid_widened.add(token_index)

    # Sanity baseline: spend the same number of token-level widening decisions as the
    # hybrid policy, but place them uniformly without rank information.
    baseline_rng = random.Random(seed + 123_456)
    random_widened = set(baseline_rng.sample(range(sequence_length), len(hybrid_widened)))

    def metrics(widened: set[int]) -> dict[str, float]:
        event_hits = 0
        for event in events:
            if any(index in widened for index in range(event.start, event.start + event.length)):
                event_hits += 1
        return {
            "event_recall": event_hits / event_count,
            "token_recall": len(attack_tokens & widened) / len(attack_tokens),
            "widen_rate": len(widened) / sequence_length,
        }

    return {
        "rank_only": metrics(rank_only_widened),
        "hybrid": metrics(hybrid_widened),
        "budget_matched_random": metrics(random_widened),
    }


def _aggregate(values: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for policy in ("rank_only", "hybrid", "budget_matched_random"):
        events = [value[policy]["event_recall"] for value in values]
        tokens = [value[policy]["token_recall"] for value in values]
        widen = [value[policy]["widen_rate"] for value in values]
        mean = statistics.mean(events)
        sd = statistics.stdev(events) if len(events) > 1 else 0.0
        half_width = 1.96 * sd / math.sqrt(len(events)) if events else 0.0
        output[policy] = {
            "event_recall_mean": mean,
            "event_recall_ci95_low": max(0.0, mean - half_width),
            "event_recall_ci95_high": min(1.0, mean + half_width),
            "token_recall_mean": statistics.mean(tokens),
            "widen_rate_mean": statistics.mean(widen),
        }
    return output


def run_benchmark(
    *,
    trials: int = DEFAULT_TRIALS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    event_count: int = DEFAULT_EVENT_COUNT,
    rate_ppm: int = DEFAULT_RATE_PPM,
) -> dict[str, object]:
    scenarios = {
        "top3_concealment": 0.0,
        "mixed_25pct_rank_triggered": 0.25,
    }
    results: dict[str, object] = {
        "benchmark_id": BENCHMARK_ID,
        "implementation_target": "v2.10.1 selection primitive",
        "trials": trials,
        "sequence_length": sequence_length,
        "events_per_trial": event_count,
        "stochastic_rate_ppm": rate_ppm,
        "base_k": BASE_K,
        "max_k": MAX_K,
        "interpretation": (
            "Synthetic mechanism validation only. Event detection means at least one token in a planted "
            "concealment span was widened. It does not measure deceptive-alignment detection."
        ),
        "scenarios": {},
    }
    scenario_results = results["scenarios"]
    assert isinstance(scenario_results, dict)
    for scenario, rank_fraction in scenarios.items():
        by_length: dict[str, object] = {}
        for event_length in DEFAULT_LENGTHS:
            values = [
                _trial(
                    sequence_length=sequence_length,
                    event_length=event_length,
                    event_count=event_count,
                    rate_ppm=rate_ppm,
                    seed=seed,
                    rank_trigger_fraction=rank_fraction,
                )
                for seed in range(1, trials + 1)
            ]
            by_length[str(event_length)] = _aggregate(values)
        scenario_results[scenario] = by_length
    return results


def _print_summary(report: dict[str, object]) -> None:
    scenarios = report["scenarios"]
    assert isinstance(scenarios, dict)
    for scenario, lengths in scenarios.items():
        print(f"\n{scenario}")
        print("length  rank-only  hybrid  random  hybrid-widen-rate")
        assert isinstance(lengths, dict)
        for length, policies in lengths.items():
            assert isinstance(policies, dict)
            rank = policies["rank_only"]["event_recall_mean"]
            hybrid = policies["hybrid"]["event_recall_mean"]
            random_value = policies["budget_matched_random"]["event_recall_mean"]
            widen = policies["hybrid"]["widen_rate_mean"]
            print(f"{int(length):>6}  {rank:>9.3f}  {hybrid:>6.3f}  {random_value:>6.3f}  {widen:>17.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic H4 hybrid-widening mechanism benchmark")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = run_benchmark(trials=args.trials)
    _print_summary(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
